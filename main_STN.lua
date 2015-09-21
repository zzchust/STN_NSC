 require('cutorch')
require('nn')
require('cunn')
require('cudnn')
require('Dataset')
--require('Training')
require('optim')
require 'image'
require 'xlua'
--require('Utilities')
torch.setnumthreads(4)
cutorch.setDevice(1)
torch.setdefaulttensortype('torch.FloatTensor')
batchSize = 128
TRAIN_DATA_PATH = '/home/zczhou/STN_NSC_RAW/data/PSB.t7'
MODEL_PATH = '/home/zczhou/STN_NSC_RAW/model'
MODEL_NAME = '/home/zczhou/STN_NSC_RAW/model/snap_epoch_006.t7'
OUTPUT = '/home/zczhou/STN_NSC_RAW/output'
TRAIN_FLAG = false
DISPLAY_RESULTS_FLAG = true
SAVE_MODEL_FLAG = true
numEpoch = 10



function loadParameters(model, params)
    p = model:parameters()
    assert(#p == #params)
    for i = 1,#p do
        p[i]:copy(params[i])
    end
end

function displayResults( trainData, model, epoch, OUTPUT)
  model:evaluate()
  epoch = epoch or 1
  OUTPUT = OUTPUT or '.'
  require 'renderMiddleMap'
  inputs = trainData:getBatch()
  predict = model:forward(inputs:cuda())
  inputs = inputs:mul(trainData.std):add(trainData.mean):byte()  
  --spanet_out = spanet.output:mul(trainData.std):add(trainData.mean):byte()
  spanet_out = model:get(1).output
  spanetImages = renderMiddleMap(spanet_out)
  inputImages = renderMiddleMap(inputs)

  filename = string.format('%s/spanet_%d.jpg', OUTPUT, epoch)
  image.save(filename, spanetImages)
  filename = string.format('%s/inputs_%d.jpg', OUTPUT, epoch)
  image.save(filename, inputImages)
end
--construct model
model = require('myAlexModel')
paths.dofile('Optim.lua')
require 'stn'
paths.dofile('spatial_transformer.lua')
model:insert(spanet,1)
model:cuda()
criterion = nn.ClassNLLCriterion():cuda()

optimState = {learningRate = 0.01, momentum = 0.9, weightDecay = 5e-4}
if  arg[1] or MODEL_NAME then
    loadModel = arg[1] or MODEL_NAME 
    params = torch.load(loadModel).parameters
    loadParameters(model, params)
    print(string.format('Model loaded from %s', loadModel))
end


optimizer = nn.Optim(model, optimState)


--load data
trainData = Dataset(batchSize)
trainData:loadDataset(TRAIN_DATA_PATH)


--training
if TRAIN_FLAG then
for epoch=1, numEpoch do
   model:training()
   local trainError = 0
   for batchId = 1, trainData.numBatch do
	   local inputs, labels = trainData:getBatch()
      err = optimizer:optimize(optim.sgd, inputs:cuda(), labels:cuda(), criterion)
      trainError = trainError + err
      if batchId % 1000 == 0 then
        collectgarbage()
      end
      xlua.progress(batchId,trainData.numBatch)
      if batchId % 50 == 0 then
         print('epoch : ', epoch, 'batchId :',batchId, 'trainError : ', trainError / batchId)
      end
   end
   print('epoch : ', epoch, 'trainError : ', trainError / trainData.numBatch)
   
   if SAVE_MODEL_FLAG then
      savePath = string.format('%s/snap_epoch_%03d.t7', MODEL_PATH, epoch)
      torch.save(savePath,
        {
          parameters = model:parameters(),
          optimState = optimState
        })
    end
   --

if DISPLAY_RESULTS_FLAG then
   displayResults( trainData, model, epoch, OUTPUT)
end

end
end

if DISPLAY_RESULTS_FLAG then
   displayResults( trainData, model, epoch, OUTPUT)
end




