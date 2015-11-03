package edu.stanford.rsl.tutorial.dmip;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.MatrixNormType;
import edu.stanford.rsl.conrad.numerics.SimpleVector.VectorNormType;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import ij.IJ;
import ij.ImageJ;
import ij.plugin.filter.Convolver;
import ij.process.FloatProcessor;


/**
 * Introduction to the CONRAD Framework
 * Exercise 1 of Diagnostic Medical Image Processing (DMIP)
 * @author Marco Boegel
 *
 */

public class Intro {
	
	
	public static void gridIntro(){
				
		//Define the image size
		int imageSizeX = 256;
		int imageSizeY = 256;
	
		//Define an image
		//Hint: Import the package edu.stanford.rsl.conrad.data.numeric.Grid2D
		//TODO
		Grid2D image = new Grid2D(imageSizeX, imageSizeY);
		//end TODO
		
		
		//Draw a circle
		int radius = 50;
		//Set all pixels within the circle to 100
		int insideVal = 100;
	
		//TODO
		//TODO	
		//TODO
		for ( int i = 0; i < imageSizeX; i ++ ) { 
			for ( int j = 0; j < imageSizeY; j ++ ) { 
				if(Math.pow(i-(imageSizeX/2.f),2) + Math.pow(j-(imageSizeY/2.f),2)<Math.pow(radius, 2)){
					image.setAtIndex(i, j, insideVal);
				}else{
					image.setAtIndex(i, j, 0);
				}
			}
		}

		//end TODO
		
		//Show ImageJ GUI
		ImageJ ij = new ImageJ();
		//Display image
		//TODO
		image.show("My image");		
		//end TODO
		
		//Copy an image
		//TODO
		Grid2D copy = new Grid2D(image);
		//Grid2D copy = (Grid2D)image.clone();
		//end TODO
		copy.show("Copy of circle");
		
		
		//Load an image from file
		String filename = "/proj/ciptmp/qe21vady/Reconstruction/CONRAD/src/edu/stanford/rsl/tutorial/dmip/mr12.dcm";
		//TODO
		//Hint: Use IJ and ImageUtil
		Grid2D mrImage = ImageUtil.wrapImagePlus(IJ.openImage(filename)).getSubGrid(0);
		//end TODO
		mrImage.show();
		
		//convolution
		//TODO
		//TODO
		Convolver conv = new Convolver();
		FloatProcessor ip = ImageUtil.wrapGrid2D(mrImage);
		
		//end TODO
		
		//define the kernel. Try simple averaging 3x3 filter
		int kw = 3;
		int kh = 3;
		float[] kernel = new float[kw*kh];
		for(int i = 0; i < kernel.length; i++)
		{	
			kernel[i] = 1.f / (kw*kh);
		}
		
		//TODO
		conv.convolve(ip, kernel, kw, kh);
		//end TODO
			
		
		//write an image to disk, check the supported output formats
		String outFilename ="/proj/ciptmp/qe21vady/Reconstruction/CONRAD/src/edu/stanford/rsl/tutorial/dmip/mr12out.tif";		
		//TODO
		//IJ.save(ImageUtil.wrapGrid2D(mrImage), outFilename); //geht so nicht
		//end TODO
	}
	
	
	public static void signalIntro()
	{
		//How can I plot a sine function sin(2*PI*x)?
		double stepSize = 0.01;
		int plotLength = 500;
		
		double[] y = new double[plotLength];
		
		for(int i = 0; i < y.length; i++)
		{
			//TODO
			y[i] = Math.sin(2.0* Math.PI * i * stepSize); 
			//end TODO
		}
		
		VisualizationUtil.createPlot(y).show(); 
		// => There will be no naming and the indices instead of the x values is shown,
		// but it looks quite good :)
		double[] x = new double [plotLength];
		for(int i = 0; i < x.length; i++)
		{
			x[i] = (double) i * stepSize;
		}
		
		VisualizationUtil.createPlot(x, y, "sin(x)", "x", "y").show();		
		
	}
	
	public static void basicIntro()
	{
		//Display text
		System.out.println("Creating a vector: v1 = [1.0; 2.0; 3.0]");
		
		//create column vector
		//TODO
		SimpleVector v1 = new SimpleVector(1.0f, 2.0f, 3.0f);
		//end TODO
		System.out.println("v1 = " + v1.toString());
		
		//create a randomly initialized vector
		SimpleVector vRand = new SimpleVector(3);
		//TODO
		vRand.randomize(0.f, 1.f);
		//end TODO
		System.out.println("vRand = " + vRand.toString());
		
		//create matrix M 3x3  1 2 3; 4 5 6; 7 8 9
		SimpleMatrix M = new SimpleMatrix(3,3);
		//TODO
		M.setColValue(0, new SimpleVector(1, 4, 7));
		M.setColValue(1, new SimpleVector(2, 5, 8));
		M.setColValue(2, new SimpleVector(3, 6, 9));
		//end TODO
		System.out.println("M = " + M.toString());
		
		//determinant of M
		System.out.println("Determinant of matrix m: " + M.determinant());
		
		//transpose M
		//TODO
		SimpleMatrix MTransposed = M.transposed();
		//end TODO
		//copy matrix
		//TODO
		SimpleMatrix MTransposedCloned = MTransposed.clone();
		// Or: SimpleMatrix MTransposedCloned = new SimpleMatrix(MTransposed)
		//end TODO
		//transpose M inplace
		//TODO
		M.transpose();
		//end TODO
		
		//get size
		int numRows = 0;
		int numCols = 0;
		//TODO
		numRows = M.getRows();
		numCols = M.getCols();
		//end TODO
		
		//access elements of M
		System.out.println("M: ");
		for(int i = 0 ; i < numRows; i++)
		{
			for(int j = 0; j < numCols; j++)
			{
				//TODO
				double element = M.getElement(i,  j);
				//end TODO
				System.out.print(element + " ");
				
			}
			System.out.println();
		}
		
		//Create 3x3 Matrix of 1's
		SimpleMatrix Mones = new SimpleMatrix(3,3);
		//TODO
		Mones.ones();
		//end TODO
		//Create a 3x3 Matrix of 0's
		SimpleMatrix Mzeros = new SimpleMatrix(3,3);
		//TODO
		Mzeros.zeros();
		//end TODO
		//Create a 3x3 Identity matrix
		SimpleMatrix Midentity = new SimpleMatrix(3,3);
		//TODO
		Midentity.identity();
		//end TODO
		
		//Matrix multiplication
		//TODO
		SimpleMatrix ResMat = SimpleOperators.multiplyMatrixProd(MTransposed, M);
		System.out.println("M^T * M = " + ResMat.toString());
		

		//Matrix vector multiplication
		//TODO
		SimpleVector resVec = SimpleOperators.multiply(M, v1);
		//end TODO
		System.out.println("M * v1 = " + resVec.toString());
		
		
		//Extract the last column vector from matrix M
		SimpleVector colVector = M.getCol(2);
		//Extract the 1x2 subvector from the last column of matrix M
		//TODO
		SimpleVector subVector = M.getSubCol(0, 2, 2); 
		//end TODO
		System.out.println("[m(0)(2); m(1)(2)] = " + subVector);
		
		//Matrix elementwise multiplication
		//TODO
		SimpleMatrix MsquaredElem = SimpleOperators.multiplyElementWise(M, M);
		//end TODO
		System.out.println("M squared Elements: " + MsquaredElem.toString());
		
		//round vectors
		SimpleVector vRandCopy = new SimpleVector(vRand);
		System.out.println("vRand         = " + vRandCopy.toString());
		
		vRandCopy.floor();
		System.out.println("vRand.floor() = " + vRandCopy.toString());
		
		vRand.ceil();
		System.out.println("vRand.ceil()  = " + vRand.toString());
		
		//min, max, mean
		double minV1 = v1.min();
		double maxV1 = v1.max();
		System.out.println("Min(v1) = " + minV1 + " Max(v1) = " + maxV1);
		
		//for matrices: iterate over row or column vectors
		SimpleVector maxVec = new SimpleVector(M.getCols());
		for(int i = 0; i < M.getCols(); i++)
		{
			maxVec.setElementValue(i, M.getCol(i).max());
		}
		double maxM = maxVec.max();
		System.out.println("Max(M) = " + maxM);
		
		
		
		//Norms
		//TODO matrix L1
		double matrixNormL1 = M.norm(MatrixNormType.MAT_NORM_FROBENIUS);
		//end TODO
		//TODO vector L2
		double vecNormL2 = colVector.norm(VectorNormType.VEC_NORM_L2);
		//end TODO
		System.out.println("||M||_F = " + matrixNormL1);
		System.out.println("||colVec||_2 = " + vecNormL2);
		
		//get normalized vector
		//TODO
		SimpleVector vecL2 = colVector.normalizedL2();
		//end TODO
		//normalize vector in-place
		//TODO
		colVector.normalizeL2();
		//end TODO
		System.out.println("Normalized colVector: " + colVector.toString());
		
		
		//SVD
		SimpleMatrix A = new SimpleMatrix(3,3);
		A.setRowValue(0, new SimpleVector(11, 10,  14));
		A.setRowValue(1, new SimpleVector(12, 11, -13));
		A.setRowValue(2, new SimpleVector(14, 13, -66));
				
		System.out.println("A = " + A.toString());
		
		//TODO SVD
		DecompositionSVD svd = new DecompositionSVD(A);
		//end TODO
		
		//print singular matrix
		System.out.println(svd.getS().toString());
		
		//get condition number
		System.out.println("Condition number of A: " + svd.cond() );
		
		//Re-compute A = U * S * V^T
		SimpleMatrix temp = SimpleOperators.multiplyMatrixProd(svd.getU(), svd.getS());
		SimpleMatrix A2 = SimpleOperators.multiplyMatrixProd(temp, svd.getV().transposed());
		System.out.println("U * S * V^T: " + A2.toString());
		
	}

	public static void main(String arg[])
	{
		basicIntro();
		gridIntro();
		signalIntro();
	}
}
