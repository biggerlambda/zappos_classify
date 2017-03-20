require "image"
require "cutorch"
require "nn"
require "cunn"
require "cudnn"

local class = require "class"
local utils = require "utils"

local cmd = torch:CmdLine()
cmd:text()
cmd:option("-im", "", "image file name")
cmd:option("-modelFile", "vggmodel.t7", "model file name")
cmd:option("-meanFile", "VGG_mean.t7", "mean file name")

params = cmd:parse(arg)

local Predict = class("predict")

function Predict:__init(modelfile, meanfile)
	self.model = torch.load(modelfile)
	print("Finished loaded model")
	self.zeromean = utils.meanSubtract(torch.load(meanfile))
end

function Predict:predict(imfile)
	local im = image.load(imfile, 3, 'byte')
	im = image.scale(im, 224, 224)
	im = im:view(1, 3, 224, 224)
	im = self.zeromean(im:double())
	return self.model:forward(im:cuda())
end

local classes = {"shoes", "boots", "sandals", "slippers"}
local p = Predict(params.modelFile, params.meanFile)
local timer = torch.Timer()
local probs = p:predict(params.im)
_, maxClass = torch.max(probs, 2)
timer:stop()
print("***********************************************")
print(string.format("Predict: %s", classes[maxClass[{1,1}]]))
print("********* Details **********")
print("Prediction time "..timer:time().real.." seconds.")
print("Classes: [shoes, boots, sandals, slippers]")
print(probs)
