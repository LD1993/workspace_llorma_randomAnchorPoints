package prea.recommender.llorma;

import prea.util.EvaluationMetrics;
import prea.util.KernelSmoothing;
import prea.recommender.Recommender;
import prea.recommender.matrix.RegularizedSVD;
import prea.data.structure.SparseVector;
import prea.data.structure.SparseMatrix;

/**
 * A class implementing Local Low-Rank Matrix Approximation.
 * Technical detail of the algorithm can be found in
 * Joonseok Lee and Seungyeon Kim and Guy Lebanon and Yoram Singer, Local Low-Rank Matrix Approximation,
 * Proceedings of the 30th International Conference on Machine Learning, 2013.
 * 
 * @author Joonseok Lee
 * @since 2013. 6. 11
 * @version 1.2
 */
public class SingletonGlobalLLORMA implements Recommender {
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
	public SparseMatrix rateMatrix;
	/** Rating matrix for items which will be used during the validation phase.
	 * (Deciding when to stop the gradient descent.)
	 * Do not directly use this during learning. */
	private SparseMatrix validationMatrix;
	/** Proportion of dataset, using for validation purpose. */
	private double validationRatio;
	
	/** The number of users. */
	public int userCount;
	/** The number of items. */
	public int itemCount;
	/** Maximum value of rating, existing in the dataset. */
	public double maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public double minValue;
	
	// Data structures:
	/** User profile in low-rank matrix form. */
	SparseMatrix[] localUserFeatures;
	/** Item profile in low-rank matrix form. */
	SparseMatrix[] localItemFeatures;
	/** Array of anchor user IDs. */
	int[] anchorUser;
	/** Array of anchor item IDs. */
	int[] anchorItem;
	/** Similarity matrix between users. */
	private static SparseMatrix userSimilarity;
	/** Similarity matrix between items. */
	private static SparseMatrix itemSimilarity;
	/** A global SVD model used for calculating user/item similarity. */
	public static RegularizedSVD baseline;
	/** Precalculated sum of weights for each user-item pair. */
	double[][] weightSum;
	
	// Local model parameters:
	/** The number of features. */
	public int featureCount;
	/** Learning rate parameter. */
	public double learningRate;
	/** Regularization factor parameter. */
	public double regularizer;
	/** Maximum number of iteration. */
	public int maxIter;
	
	// Global combination parameters:
	/** Maximum number of local models. */
	public int modelMax;
	/** Type of kernel used in kernel smoothing. */
	public int kernelType;
	/** Width of kernel used in kernel smoothing. */
	public double kernelWidth;
	
	/**Width of kernel used for disease kernel smoothing*/
	public double kernelWidth1 = 2;
	
	/** Indicator whether to show progress of iteration. */
	public boolean showProgress;
	
	/** The matrices holds drug similarity*/
	public static Double[][] drugSim;
	/** The matrices holds drug similarity*/
	public static Double[][] diseaseSim;
	/**Maps drug name to drug id and back */
	
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a matrix-factorization-based model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 * @param fc The number of features used for describing user and item profiles.
	 * @param lr Learning rate for gradient-based or iterative optimization.
	 * @param r Controlling factor for the degree of regularization. 
	 * @param iter The maximum number of iterations.
	 * @param mm The maximum number of local models.
	 * @param kt Type of kernel used in kernel smoothing.
	 * @param kw Width of kernel used in kernel smoothing.
	 * @param vr The ratio of training set which will be used for validation purpose.
	 * @param base A global SVD model used for calculating user/item similarity.
	 * @param verbose Indicating whether to show iteration steps and train error.
	 */
	public SingletonGlobalLLORMA(int uc, int ic, double max, double min, int fc,
			double lr, double r, int iter, int mm, int kt, double kw, double vr,
			RegularizedSVD base, boolean verbose) {
		userCount = uc;
		itemCount = ic;
		maxValue = max;
		//minValue = min;
		minValue = 0;
		
		featureCount = fc;
		learningRate = lr;
		regularizer = r;
		maxIter = iter;
		
		modelMax = mm;
		kernelType = kt;
		kernelWidth = kw;
		validationRatio = vr;
		baseline = base;
		
		showProgress = verbose;
	}
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a matrix-factorization-based model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 * @param fc The number of features used for describing user and item profiles.
	 * @param lr Learning rate for gradient-based or iterative optimization.
	 * @param r Controlling factor for the degree of regularization. 
	 * @param iter The maximum number of iterations.
	 * @param mm The maximum number of local models.
	 * @param kt Type of kernel used in kernel smoothing.
	 * @param kw Width of kernel used in kernel smoothing.
	 * @param vr The ratio of training set which will be used for validation purpose.
	 * @param base A global SVD model used for calculating user/item similarity.
	 * @param verbose Indicating whether to show iteration steps and train error.
	 * @param drugSim is the chemicalStructureBased similarity matrix for drugs
	 * @param diseaseSim is the symptomsBased similarity matrix for diseases
	 */
	public SingletonGlobalLLORMA(int uc, int ic, double max, double min, int fc,
			double lr, double r, int iter, int mm, int kt, double kw, double vr,
			RegularizedSVD base, boolean verbose, Double[][] drugSim, Double[][] diseaseSim) {
		userCount = uc;
		itemCount = ic;
		maxValue = max;
		minValue = min;
		
		featureCount = fc;
		learningRate = lr;
		regularizer = r;
		maxIter = iter;
		
		modelMax = mm;
		kernelType = kt;
		kernelWidth = kw;
		validationRatio = vr;
		baseline = base;
		
		showProgress = verbose;
		this.drugSim = drugSim;
		this.diseaseSim =  diseaseSim;			
		
		/*
		for(int i=0; i< 1240; i++){
			for(int j=0; j< 1240; j++){
				System.out.println(this.drugSim[i][j]);
			}
		}
		
		for(int i=0; i< 300; i++){
			for(int j=0; j< 300; j++){
				System.out.println(this.diseaseSim[i][j]);
			}
		}
		System.out.println();
		*/
		
	}
	
	/*========================================
	 * Get Drug Sim Matrix Approximation
	 =========================================*/
	/**
	 * Sudipta : 
	 * 
	 * @param 
	 */
	private double getFrobeniusNorm_Drug(SparseMatrix userFeature){
		double fNorm = 0.0;
		for(int rowIdx1=0; rowIdx1< userCount; rowIdx1++){
			for(int rowIdx2=rowIdx1; rowIdx2< userCount; rowIdx2++){
				double innerPdtEntry = userFeature.getRow(rowIdx1).innerProduct(userFeature.getRow(rowIdx2));
				fNorm += ((drugSim[rowIdx1][rowIdx2]-innerPdtEntry)*(drugSim[rowIdx1][rowIdx2]-innerPdtEntry));
			}
		}
		fNorm = Math.sqrt(fNorm);
		return fNorm;
	}
	
	/*========================================
	 * Get Disease Sim Matrix Approximation
	 =========================================*/
	/**
	 * Sudipta : 
	 * 
	 * @param 
	 */
	private double getFrobeniusNorm_Disease(SparseMatrix itemFeature){
		double fNorm = 0.0;
		for(int rowIdx1=0; rowIdx1< itemCount; rowIdx1++){
			for(int rowIdx2=0; rowIdx2< itemCount; rowIdx2++){
				double innerPdtEntry = itemFeature.getRow(rowIdx1).innerProduct(itemFeature.getRow(rowIdx2));
				fNorm += ((diseaseSim[rowIdx1][rowIdx2]-innerPdtEntry)*(diseaseSim[rowIdx1][rowIdx2]-innerPdtEntry));
			}
		}
		fNorm = Math.sqrt(fNorm);
		return fNorm;
	}
	
	/*========================================
	 * Model Builder
	 *========================================*/
	/**
	 * Build a model with given training set.
	 * 
	 * Sudipta : Change
	 * 
	 * @param rateMatrix The rating matrix with train data.
	 */
	@Override
	public void buildModel(SparseMatrix rateMatrix) {
		makeValidationSet(rateMatrix, validationRatio);
		
		// Preparing data structures:
		localUserFeatures = new SparseMatrix[modelMax];
		localItemFeatures = new SparseMatrix[modelMax];
		
		anchorUser = new int[modelMax];
		anchorItem = new int[modelMax];
		
		for (int l = 0; l < modelMax; l++) {
			boolean done = false;
			while (!done) {
				int u_t = (int) Math.floor(Math.random() * userCount) + 1;
				int[] itemList = rateMatrix.getRow(u_t).indexList();
	
				if (itemList != null) {
					int idx = (int) Math.floor(Math.random() * itemList.length);
					int i_t = itemList[idx];
					
					anchorUser[l] = u_t;
					anchorItem[l] = i_t;
					
					done = true;
				}
			}
		}
		
		// Pre-calculating similarity:
		userSimilarity = new SparseMatrix(userCount+1, userCount+1);
		itemSimilarity = new SparseMatrix(itemCount+1, itemCount+1);
		
		weightSum = new double[userCount+1][itemCount+1];
		for (int u = 1; u <= userCount; u++) {
			for (int i = 1; i <= itemCount; i++) {
				for (int l = 0; l < modelMax; l++) {
					/*weightSum[u][i] += KernelSmoothing.kernelize(getUserSimilarity(anchorUser[l], u), kernelWidth, kernelType)
									 * KernelSmoothing.kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType);*/
					double drugSim = getChemicalStructureBasedDrugSimilarity(anchorUser[l], u);
					double drugKernel = KernelSmoothing.kernelize(drugSim, kernelWidth, kernelType);
					
					double diseaseSim = getClinicalSymptomBasedDiseaseSimilarity(anchorItem[l], i);
					double diseaseKernel = KernelSmoothing.kernelize(diseaseSim, kernelWidth1, kernelType);
					if(drugKernel==0){
						System.out.println(diseaseKernel);
					}
					if(diseaseKernel==0){
						System.out.println(diseaseKernel);
					}
					double nWeight = drugKernel * diseaseKernel;
					
					weightSum[u][i] += nWeight;					
				}
				if(weightSum[u][i]==0.0){
					System.out.println(weightSum[u][i]);
				}
			}
		}
		
		// Initialize local models:
		for (int l = 0; l < modelMax; l++) {
			localUserFeatures[l] = new SparseMatrix(userCount+1, featureCount);
			localItemFeatures[l] = new SparseMatrix(featureCount, itemCount+1);
			
			for (int u = 1; u <= userCount; u++) {
				for (int r = 0; r < featureCount; r++) {
					double rdm = Math.random();
					localUserFeatures[l].setValue(u, r, rdm);
				}
			}
			for (int i = 1; i <= itemCount; i++) {
				for (int r = 0; r < featureCount; r++) {
					double rdm = Math.random();
					localItemFeatures[l].setValue(r, i, rdm);
				}
			}
		}
		
		// Learn by gradient descent:
		int round = 0;
		double prevErr = 99999;
		double currErr = 9999;
		
		while (Math.abs(prevErr - currErr) > 0.001 && round < maxIter) {
			EvaluationMetrics e = this.evaluate(validationMatrix);
			prevErr = currErr;
			currErr = e.getRMSE();
			
			// Show progress:
			if (showProgress) {
				System.out.println(round + "\t"	+ e.printOneLine());
			}
			
			double fNomeSum1 = 0.0;
			double fNomeSum2 = 0.0;
			
			for (int u = 1; u <= userCount; u++) {
				fNomeSum1 = 0.0;
				fNomeSum2 = 0.0;
				SparseVector items = rateMatrix.getRowRef(u);
				int[] itemIndexList = items.indexList();
				
				if (itemIndexList != null) {
					for (int i : itemIndexList) {
						// current estimation:
						double RuiEst = 0.0;
						for (int l = 0; l < modelMax; l++) {
							/*RuiEst += localUserFeatures[l].getRow(u).innerProduct(localItemFeatures[l].getCol(i))
									* KernelSmoothing.kernelize(getUserSimilarity(anchorUser[l], u), kernelWidth, kernelType)
								 	* KernelSmoothing.kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType)
								 	/ weightSum[u][i];*/
							/*if(localUserFeatures[l].getRow(u).innerProduct(localItemFeatures[l].getCol(i))<1.0){
								System.out.println(localUserFeatures[l].getRow(u));
								System.out.println(localItemFeatures[l].getCol(i));
							}*/
							RuiEst += localUserFeatures[l].getRow(u).innerProduct(localItemFeatures[l].getCol(i))
									* KernelSmoothing.kernelize(getChemicalStructureBasedDrugSimilarity(anchorUser[l], u), kernelWidth, kernelType)
								 	* KernelSmoothing.kernelize(getClinicalSymptomBasedDiseaseSimilarity(anchorItem[l], i), kernelWidth1, kernelType)
								 	/ weightSum[u][i];
							
							/*
							double fNorm1 = 0.0;
							//for(int rowIdx1=0; rowIdx1< userCount; rowIdx1++){
								for(int rowIdx2=0; rowIdx2< userCount; rowIdx2++){
									//fNorm1 = 0.0;
									if(u==1 && rowIdx2==1){
										//System.out.println();
									}
									//System.out.println("localUserFeatures[l].getRow(u) : " + localUserFeatures[l].getRow(u));
									//System.out.println("localUserFeatures[l].getRow(rowIdx2) : " + localUserFeatures[l].getRow(u));
									double innerPdtEntry = localUserFeatures[l].getRow(u).innerProduct(localUserFeatures[l].getRow(rowIdx2))/featureCount;
									
									double d1 = ((drugSim[u-1][rowIdx2]-innerPdtEntry)*(drugSim[u-1][rowIdx2]-innerPdtEntry));
									if(Double.isInfinite(d1) || Double.isNaN(d1)){
										System.out.println();
									}
									
									fNorm1 += ((drugSim[u-1][rowIdx2]-innerPdtEntry)*(drugSim[u-1][rowIdx2]-innerPdtEntry));
								}
							//}
							fNomeSum1 += (Math.sqrt(fNorm1))/modelMax;
							if(Double.isInfinite(fNomeSum1) || Double.isNaN(fNomeSum1)){
								System.out.println("fNomeSum1 Infinite or NaN");
								System.exit(1);
							}
							
							double fNorm2 = 0.0;
							//for(int colIdx1=0; colIdx1< itemIndexList.length; colIdx1++){
								for(int colIdx2=0; colIdx2< itemCount; colIdx2++){
									//fNorm2=0.0;
									//double innerPdtEntry = localItemFeatures[l].getCol(itemIndexList[colIdx1]-1).innerProduct(localItemFeatures[l].getCol(colIdx2))/featureCount;
									double innerPdtEntry = localItemFeatures[l].getCol(i).innerProduct(localItemFeatures[l].getCol(colIdx2))/featureCount;
									//double d2 = ((diseaseSim[itemIndexList[colIdx1]-1][colIdx2]-innerPdtEntry)*(diseaseSim[itemIndexList[colIdx1]-1][colIdx2]-innerPdtEntry));
									double d2 = ((diseaseSim[i-1][colIdx2]-innerPdtEntry)*(diseaseSim[i-1][colIdx2]-innerPdtEntry));
									if(Double.isInfinite(d2) || Double.isNaN(d2)){
										System.out.println();
									}
									fNorm2 += ((diseaseSim[i-1][colIdx2]-innerPdtEntry)*(diseaseSim[i-1][colIdx2]-innerPdtEntry));
									//fNorm2 += ((diseaseSim[itemIndexList[colIdx1]-1][colIdx2]-innerPdtEntry)*(diseaseSim[itemIndexList[colIdx1]-1][colIdx2]-innerPdtEntry));
								}
							//}
							fNomeSum2 += (Math.sqrt(fNorm2))/modelMax;
							if(Double.isInfinite(fNomeSum2) || Double.isNaN(fNomeSum2)){
								System.out.println("fNomeSum2 Infinite or NaN");
								System.exit(1);
							}*/
							
							
						}
						double RuiReal = rateMatrix.getValue(u, i);
						//double err = (RuiReal - RuiEst)+fNomeSum1+fNomeSum2;
						double err = (RuiReal - RuiEst);
						
						if(fNomeSum1>1 ){
							//System.out.println("fNomeSum1 Error : " + fNomeSum1);
							//System.exit(1);
						}
						if(fNomeSum2>1 ){
							//System.out.println("fNomeSum2 Error : "   + fNomeSum2);
							//System.exit(1);
						}
						fNomeSum1 = 0;
						fNomeSum2= 0;
						//double err = (RuiReal - RuiEst);
						
						
						// parameter update:
						for (int l = 0; l < modelMax; l++) {
							/*double weight = KernelSmoothing.kernelize(getUserSimilarity(anchorUser[l], u), kernelWidth, kernelType)
									 	  * KernelSmoothing.kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType)
									 	  / weightSum[u][i];*/
							double weight = KernelSmoothing.kernelize(getChemicalStructureBasedDrugSimilarity(anchorUser[l], u), kernelWidth, kernelType)
								 	  * KernelSmoothing.kernelize(getClinicalSymptomBasedDiseaseSimilarity(anchorItem[l], i), kernelWidth1, kernelType)
								 	  / weightSum[u][i];
							
							for (int r = 0; r < featureCount; r++) {
								double Fus = localUserFeatures[l].getValue(u, r);
								double Gis = localItemFeatures[l].getValue(r, i);	
								
								if((err*Gis*weight - regularizer*Fus)<0){
									//System.out.println("Terminating");
									//System.exit(1);
								}
								
								if((err*Fus*weight - regularizer*Gis)<0){
									//System.out.println("Terminating");
									//System.exit(1);
								}
								
								localUserFeatures[l].setValue(u, r, Fus + learningRate*(err*Gis*weight - regularizer*Fus));
								if(Double.isNaN(Fus + learningRate*(err*Gis*weight - regularizer*Fus))) {
									System.out.println("a");
								}
								localItemFeatures[l].setValue(r, i, Gis + learningRate*(err*Fus*weight - regularizer*Gis));
								if(Double.isNaN(Gis + learningRate*(err*Fus*weight - regularizer*Gis))) {
									System.out.println("b");
								}
							}
						}
					}
				}
			}
			
			round++;
			System.out.println("round # : " + round);
		}
		
		restoreValidationSet(rateMatrix);
	}
	
	
	/*========================================
	 * Prediction
	 *========================================*/
	/**
	 * Evaluate the designated algorithm with the given test data.
	 * 
	 * @param testMatrix The rating matrix with test data.
	 * 
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	@Override
	public EvaluationMetrics evaluate(SparseMatrix testMatrix) {
		SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
		
		for (int u = 1; u <= userCount; u++) {
			SparseVector items = testMatrix.getRowRef(u);
			int[] itemIndexList = items.indexList();
			
			if (itemIndexList != null) {
				for (int i : itemIndexList) {
					double prediction = 0.0;
					for (int l = 0; l < modelMax; l++) {
						/*prediction += localUserFeatures[l].getRow(u).innerProduct(localItemFeatures[l].getCol(i))
								* KernelSmoothing.kernelize(getUserSimilarity(anchorUser[l], u), kernelWidth, kernelType)
							 	* KernelSmoothing.kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType)
							 	/ weightSum[u][i];*/
						//System.out.println("localUserFeatures[l].getRow(u) " + localUserFeatures[l].getRow(u));
						//System.out.println("localItemFeatures[l].getCol(i) " + localItemFeatures[l].getCol(i));
						//System.out.println("localUserFeatures[l].getRow(u).innerProduct(localItemFeatures[l].getCol(i)) = " + localUserFeatures[l].getRow(u).innerProduct(localItemFeatures[l].getCol(i)));
						//System.out.println("KernelSmoothing.kernelize(getChemicalStructureBasedDrugSimilarity(anchorUser[l], u), kernelWidth, kernelType) = " + KernelSmoothing.kernelize(getChemicalStructureBasedDrugSimilarity(anchorUser[l], u), kernelWidth, kernelType));
						//System.out.println("KernelSmoothing.kernelize(getClinicalSymptomBasedDiseaseSimilarity(anchorItem[l], i), kernelWidth1, kernelType) = " + KernelSmoothing.kernelize(getClinicalSymptomBasedDiseaseSimilarity(anchorItem[l], i), kernelWidth1, kernelType));
						//System.out.println("weightSum[u][i] = " + weightSum[u][i]);						
						
						prediction += localUserFeatures[l].getRow(u).innerProduct(localItemFeatures[l].getCol(i))
								* KernelSmoothing.kernelize(getChemicalStructureBasedDrugSimilarity(anchorUser[l], u), kernelWidth, kernelType)
							 	* KernelSmoothing.kernelize(getClinicalSymptomBasedDiseaseSimilarity(anchorItem[l], i), kernelWidth1, kernelType)
							 	/ weightSum[u][i];
					}
					
					if(prediction>1.0){
						//System.out.println("prediction = " + prediction);
					}
					
					if (Double.isNaN(prediction) || prediction == 0.0) {
						prediction = (maxValue + minValue)/2;
					}
					
					if (prediction < minValue) {
					//if (prediction < 1) {
						//prediction = minValue;
					}
					else if (prediction > maxValue) {
					//else if (prediction > 2) {
						//prediction = maxValue;
					}
					
					predicted.setValue(u, i, prediction);
				}
			}
		}
		
		return new EvaluationMetrics(testMatrix, predicted, maxValue, minValue);
	}
	
	
	/**
	 * Items which will be used for validation purpose are moved from rateMatrix to validationMatrix.
	 * 
	 * @param validationRatio Proportion of dataset, using for validation purpose.
	 */
	private void makeValidationSet(SparseMatrix rateMatrix, double validationRatio) {
		validationMatrix = new SparseMatrix(userCount+1, itemCount+1);
		
		int validationCount = (int) (rateMatrix.itemCount() * validationRatio);
		while (validationCount > 0) {
			int index = (int) (Math.random() * userCount) + 1;
			SparseVector row = rateMatrix.getRowRef(index);
			int[] itemList = row.indexList();
			
			if (itemList != null && itemList.length > 5) {
				int index2 = (int) (Math.random() * itemList.length);
				validationMatrix.setValue(index, itemList[index2], rateMatrix.getValue(index, itemList[index2]));
				rateMatrix.setValue(index, itemList[index2], 0.0);
				
				validationCount--;
			}
		}
	}
	
	/** Items in validationMatrix are moved to original rateMatrix. */
	private void restoreValidationSet(SparseMatrix rateMatrix) {
		for (int i = 1; i <= userCount; i++) {
			SparseVector row = validationMatrix.getRowRef(i);
			int[] itemList = row.indexList();
			
			if (itemList != null) {
				for (int j : itemList) {
					rateMatrix.setValue(i, j, validationMatrix.getValue(i, j));
				}
			}
		}
	}
	
	/**
	 * Calculate similarity between two drugs, based on the chemical structure based drug similarity matrix provided.
	 * 
	 * @param idx1 The first drug's ID.
	 * @param idx2 The second drug's ID.
	 * 
	 * @return The similarity value between two drugs idx1 and idx2.
	 */
	private static double getChemicalStructureBasedDrugSimilarity (int idx1, int idx2) {
		double sim = -1;
		try{
			if(idx1<1 || idx2<1){
				System.out.println("index value incorrect");
				System.exit(0);
			}else{	
				idx1--;
				idx2--;
				//if(drugSim[idx1][idx2]!=drugSim[idx2][idx1]){
					//System.out.println("Drug similarity matrix needs to be symmetric");
					//System.exit(0);
				//}else{
					sim = drugSim[idx1][idx2];
				//}			
			}
		}catch(Exception e){
			System.out.println(e.getMessage());
		}
		return sim;
	}
	
	/**
	 * Calculate similarity between two users, based on the global base SVD.
	 * 
	 * @param idx1 The first user's ID.
	 * @param idx2 The second user's ID.
	 * 
	 * @return The similarity value between two users idx1 and idx2.
	 */
	private static double getUserSimilarity (int idx1, int idx2) {
		double sim;
		if (idx1 <= idx2) {
			sim = userSimilarity.getValue(idx1, idx2);
		}
		else {
			sim = userSimilarity.getValue(idx2, idx1);
		}
		
		if (sim == 0.0) {
			SparseVector u_vec = baseline.getU().getRowRef(idx1);
			SparseVector v_vec = baseline.getU().getRowRef(idx2);
			
			sim = 1 - 2.0 / Math.PI * Math.acos(u_vec.innerProduct(v_vec) / (u_vec.norm() * v_vec.norm()));
			
			if (Double.isNaN(sim)) {
				sim = 0.0;
			}
			
			if (idx1 <= idx2) {
				userSimilarity.setValue(idx1, idx2, sim);
			}
			else {
				userSimilarity.setValue(idx2, idx1, sim);
			}
		}
		
		return sim;
	}
	
	/**
	 * Calculate similarity between two diseases, based on the clinical symptoms based similarity matrix.
	 * 
	 * @param idx1 The first disease's ID.
	 * @param idx2 The second disease's ID.
	 * 
	 * @return The similarity value between two diseases idx1 and idx2.
	 * 
	 * Comment : Index values should be reduced by one before using
	 */
	private static double getClinicalSymptomBasedDiseaseSimilarity (int idx1, int idx2) {
		double sim = -1;
		try{
			if(idx1<1 || idx2<1){
				System.out.println("index value incorrect");
				System.exit(0);
			}else{	
				idx1--;
				idx2--;
				//if(diseaseSim[idx1][idx2]!=diseaseSim[idx2][idx1]){
					//System.out.println("Disease similarity matrix needs to be symmetric");
					//System.exit(0);
				//}else{
					sim = diseaseSim[idx1][idx2];
				//}			
			}
		}catch(Exception e){
			System.out.println(e.getMessage());
		}
		return sim;
	}
	
	/**
	 * Calculate similarity between two items, based on the global base SVD.
	 * 
	 * @param idx1 The first item's ID.
	 * @param idx2 The second item's ID.
	 * 
	 * @return The similarity value between two items idx1 and idx2.
	 */
	private static double getItemSimilarity (int idx1, int idx2) {
		double sim;
		if (idx1 <= idx2) {
			sim = itemSimilarity.getValue(idx1, idx2);
		}
		else {
			sim = itemSimilarity.getValue(idx2, idx1);
		} 
		
		if (sim == 0.0) {
			SparseVector i_vec = baseline.getV().getColRef(idx1);
			SparseVector j_vec = baseline.getV().getColRef(idx2);
			
			sim = 1 - 2.0 / Math.PI * Math.acos(i_vec.innerProduct(j_vec) / (i_vec.norm() * j_vec.norm()));
			
			if (Double.isNaN(sim)) {
				sim = 0.0;
			}
			
			if (idx1 <= idx2) {
				itemSimilarity.setValue(idx1, idx2, sim);
			}
			else {
				itemSimilarity.setValue(idx2, idx1, sim);
			}
		}
		
		return sim;
	}
}
