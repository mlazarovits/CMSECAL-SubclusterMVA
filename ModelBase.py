from abc import ABC, abstractmethod

class ModelBase(ABC):
	def __init__(self):
		self._model = None
		super().__init__()
	
	@abstractmethod
	def BuildModel(self):
		pass

	@abstractmethod
	def CompileModel(self):
		pass

	def TrainModel(self,x,y,epochs = 1,viz:bool = False,verb= 0):
		his = self._model.fit(x,y,epochs=epochs,verbose=verb)
		if viz:
			VizLoss(his)

	#for dataset made from generator
	#needs to be able to save (output) and load (input) weights
	def TrainModel(self,train_dataset,epochs=1,batch_size=1,weights_dir="./weights/",viz:bool = False,verb=1, val_data = None, fname='fig.pdf'):
		checkpt_dir = self.SaveWeights(batch_size, weights_dir)
		train_dataset = train_dataset.batch(batch_size)
		if val_data is not None:
			print("with val data")
			val_data = val_data.batch(batch_size)
			his = self._model.fit(train_dataset,epochs=epochs,batch_size=batch_size,verbose=verb, validation_data = val_data,callbacks=[self.cp_callback])
		else:
			his = self._model.fit(train_dataset,epochs=epochs,batch_size=batch_size,verbose=verb,callbacks=[self.cp_callback])
		if viz:
			self.VizLoss(his,fname)
		return checkpt_dir

	#load (input) weights
	def LoadWeights(self,checkpt_dir):
		print("Training network with latest weights from",checkpt_dir)
		latest = train.latest_checkpoint(checkpt_dir) 
		self._model.load_weights(latest)
		return
	
	#save (output) weights from current model
	def SaveWeights(self, batch_size, weights_dir):
		#save weights in nnue_training_weights
		checkpt_path = weights_dir+"/cp-{epoch:04d}.ckpt"
		checkpt_dir = os.path.dirname(checkpt_path)
		
		#create callback to save weights every epoch
		self.cp_callback = callbacks.ModelCheckpoint(
			filepath=checkpt_path,
			verbose=1,
			save_weights_only=True,
			save_freq = batch_size)
			#save_freq = 5*batch_size)
		self._model.save_weights(checkpt_path.format(epoch=0))
		print("Directory with model checkpoint is:",checkpt_dir)
		return checkpt_dir

	def GetModel(self):
		if self._model is None:
			print("Model not built yet")
			return
		return self._model

	def Predict(self, data, batch_size):
		#for singular instance (not from data generator)
		if type(data) is tuple:
			y_pred = self._model.predict(data,batch_size=1)
			###ply is corresponding ply number for each move (instance) in data dataset
			y_pred_cp = self.nn_to_winrate(y_pred,batch_size)	

		else:
			plies = []
			evals = []
			for element in data.as_numpy_iterator():
				evals.append(element[1])
				plies.append(element[2])
			data = data.batch(batch_size)
			y_pred = self._model.predict(data,batch_size)
			##ply is corresponding ply number for each move (instance) in data dataset
			y_pred_cp = [self.nn_to_winrate(m,plies[i]) for i, m in enumerate(y_pred)]	
		return y_pred_cp

	#to get metrics for test data
	def Evaluate(self, data, batch_size):
		y_pred = self._model.predict(data,batch_size)
		#mse loss - could probably use built-in TF functionality for this
		loss = [(true - pred)**2 for true, pred in zip(evals,y_pred)]	
		return y_pred, loss


	def VizLoss(self, history, fname):
		plt.figure()
		plt.plot(history.history['val_loss'], label="val loss")
		plt.plot(history.history['loss'],label="train loss")
		plt.title("Loss")
		plt.legend()
		plt.savefig("loss/"+fname)



