local utils = require "utils"

local getLabels = function(classSizes)
	-- construct label tensor out of table of class sizes
	local labels = torch.Tensor(utils.sum(classSizes))
	local index = 1
	for i = 1, #classSizes do
		labels[{{index, index + classSizes[i] - 1}}] = i * torch.ones(classSizes[i])
		index = index + classSizes[i]
	end
	return labels
end


local cat = function(A, B)
	-- just in case we are appending a to a. 
	local bLen = #B
	for i = 1, bLen do
		A[#A + 1] = B[i]
	end
	return A
end

function split(classFileList, trainperc, testperc, oversample)
	local trainFiles = {}
	local testFiles = {}
	local validFiles = {}

	local train = {}
	local test = {}
	local valid = {}

	local classes = {}
	local labels = {}
	for i = 1, #classFileList do
		classes[i] = utils.loadFile(classFileList[i])
		labels[i] = i
	end
	-- check oversampling sizes
	local maxSize = #classes[1]
	for i = 2, #classes do
		maxSize = math.max(maxSize, #classes[i])
	end
	
	local ratios = {}
	for i = 1, #classes do
		ratios[i] = math.ceil(maxSize / #classes[i])
	end

	local train = {}
	local test = {}
	local valid = {}

	local trainLSizes = {}
	local testLSizes = {}
	local validLSizes = {}

	for i = 1, #classes do
		local classTrain = {}
		local classTest = {}
		local classValid = {}

		local trainSplit = math.floor(trainperc * #classes[i])
		local testSplit = math.floor((trainperc + testperc) * #classes[i])

		classTrain = cat(classTrain, utils.subrange(classes[i], 1, trainSplit))
		--Oversample
		print(string.format("Oversampling class %d of size %d by %d", i, #classes[i], ratios[i]))
		local temp = utils.clone(classTrain)
		for j = 1, ratios[i] - 1  do
			classTrain = cat(classTrain, temp)
		end
		classTest = cat(classTest, utils.subrange(classes[i], trainSplit + 1, testSplit))
		classValid = cat(classValid, utils.subrange(classes[i], testSplit, #classes[i]))
		-- set up labels
		print(string.format("train %d test %d valid %d", #classTrain, #classTest, #classValid))
		trainLSizes[i] = #classTrain
		testLSizes[i] = #classTest
		validLSizes[i] = #classValid

		train = cat(train, classTrain)
		test = cat(test, classTest)
		valid = cat(valid, classValid)
	end

	local trainL = getLabels(trainLSizes)
	local testL = getLabels(testLSizes)
	local validL = getLabels(validLSizes)
	return {train, trainL}, {test, testL}, {valid, validL}
end

function loadBatch(imageList)
	local images = torch.ByteTensor(#imageList, 3, 224, 224)
	for i = 1, #imageList do
		local im = image.load(imageList[i], 3, 'byte')
		images[i] = image.scale(im, 224, 224)
	end

	return images
end
