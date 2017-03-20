require "image"
require "loadcaffe"
require "io"
require "nn"
require "nnlr"
require "optim"
require "cunn"
require "cutorch"
require "data"
local utils = require "utils"

cmd = torch:CmdLine()
cmd:text()
cmd:option("-batchSize", "100", "BatchSize")
cmd:option("-epoch", "5", "epochs")
cmd:option("-lr", "1e-3", "learning rate")

-- parse input params
params = cmd:parse(arg)

-- load data for all the four classes. Do train, test separation.
local imW = 224
local imH = 224
local shoes = "shoes.txt"
local boots = "boots.txt"
local sandals = "sandals.txt"
local slippers = "slippers.txt"

local classFileList = {shoes, boots, sandals, slippers}
local classes = {1, 2, 3, 4}

local trainperc = 0.85
local validperc = 0.05
local testperc = 0.10

local trainset = nil
local testset = nil
local validset = nil

trainset, testset, validset = split(classFileList, trainperc, testperc, true)
local train = trainset[1]
local trainL = trainset[2]

local test = testset[1]
local testL = testset[2]

local valid = validset[1]
local validL = validset[2]

print("Train size: "..#train..". Test size: "..#test..". Valid size: "..#valid)

-- load trained model
print("Loading vgg model")
local vggmodel = loadcaffe.load("VGG_CNN_M_deploy.prototxt", "VGG_CNN_M.caffemodel", "cudnn")
-- delete last 2 layers of vgg
vggmodel:remove(24)
vggmodel:remove(23)

-- add linear layer 
vggmodel:add(nn.Linear(4096, 4):cuda())
vggmodel:add(nn.LogSoftMax():cuda())

--for optim tie down learning rate for all layers except for the last 2 to 1e-3
vggmodel:get(1):learningRate("weight", 1e-3)
vggmodel:get(5):learningRate("weight", 1e-3)
vggmodel:get(9):learningRate("weight", 1e-3)
vggmodel:get(11):learningRate("weight", 1e-3)
vggmodel:get(13):learningRate("weight", 1e-3)
vggmodel:get(17):learningRate("weight", 1e-3)
vggmodel:get(20):learningRate("weight", 1e-3)
vggmodel:get(23):learningRate("weight", 1)

local baseLearningRate = params.lr
local baseWeightDecay = 1e-3
local learningRates, weightDecays = vggmodel:getOptimConfig(baseLearningRate, baseWeightDecay)
	
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()
-- Load mean
print("Zero meaning the inputs")
local mean = torch.load("VGG_mean.t7")
local zeromean = utils.meanSubtract(mean)

print("Starting training")
torch.manualSeed(1)

for epoch = 1, params.epoch do
	local modelName =string.format("model_epoch%d.t7", epoch)
	-- check if model file exists. Then skip this epoch
	if utils.file_exists(modelName) then 
		print(string.format("Found model %s. Skipping epoch %d", modelName, epoch))
		vggmodel = torch.load(modelName)
	else
		print(string.format("Training for epoch %d", epoch))
		local shuff_index = torch.randperm(#train)
		for i = 1, #train, params.batchSize do
				local j = shuff_index[i]
				local endInd = math.min(j + params.batchSize - 1, #train)
				local batch = loadBatch(utils.subrange(train, j, endInd))
				local inputs = zeromean(batch:double()):cuda()
				local labels = trainL[{{j, endInd}}]:double():clone():cuda()
					
				local weights, grad = vggmodel:getParameters()
				local feval = function(x)
					collectgarbage()
					if x ~= weights then
						weights:copy(x)
					end
					grad:zero()
					local outputs = vggmodel:forward(inputs)
					local f = criterion:forward(outputs, labels)
					local df_dw = criterion:backward(outputs, labels)
					vggmodel:backward(inputs, df_dw)
					
					grad:mul(1/inputs:size(1))
					return f, grad
				end
				optim.sgd(feval, weights, {
					learningRates = learningRates,
					weightDecays = weightDecays,
					learningRate = baseLearningRate,
					momentum = 0.9,
				})
				xlua.progress(i, #train)
		end
		--save model after each iteration
		vggmodel = vggmodel:clearState()
		torch.save(modelName, vggmodel)
		-- print validation set results at end of epoch
		collectgarbage()
		local loss = 0
		for v = 1, #valid, 20 do
			local endIndex = math.min(#valid, v + 20 - 1)
			local batch = loadBatch(utils.subrange(valid, v, endIndex))
			local outputs = vggmodel:forward(zeromean(batch:double()):cuda())
			loss = loss +  criterion:forward(outputs, validL[{{v, endIndex}}]:clone():cuda())
		end
		print(string.format("loss after epoch %d is %f", epoch, loss))
	end
end
collectgarbage()

print("Done training. Running on test")
local classes = {'1', '2', '3', '4'}
local confusion = optim.ConfusionMatrix(classes)
local outputs = torch.Tensor(#test, #classes):cuda()

for v = 1, #test, 20 do
	local endIndex = math.min(#test, v + 19)
	local batch = loadBatch(utils.subrange(test, v, endIndex))
	outputs[{{v, endIndex}}] = vggmodel:forward(zeromean(batch:double()):cuda())
end

testL = testL:cuda()
for i = 1, #test do
	confusion:add(outputs[i], testL[i])
end

print(confusion)

print("Saving model")
vggmodel = vggmodel:clearState()
torch.save("vggmodel.t7", vggmodel)
