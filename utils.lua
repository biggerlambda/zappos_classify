local ut = {}

function ut.meanSubtract(mean)
	local meanFunc = function(A)
		-- A's shape is batchSize x channels x height x with. mean is 1 x channels x height x width
		return A:csub(mean:expandAs(A))
	end

	return meanFunc
end

function ut.file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function ut.clone(t)
	s = {}
	for key, value in pairs(t) do
		s[key] = value
	end

	return s
end

function ut.sum(t)
	local s = 0
	for i = 1, #t do
		s = s + t[i]
	end
	return s
end

function ut.loadFile(filename)
	-- return a table with every line in file as a value
	local lines = {}
	for line in io.lines(filename) do
		lines[#lines +1] = line
	end
	return lines
end

function ut.subrange(A, startI, endI)
	local sub = {}
	for i = startI, endI do
		sub[#sub + 1] = A[i]
	end
	return sub
end
return ut
