package insurance;

import java.util.Scanner;

import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class InsurancePrediction {
	/**
   	* The main method accepts a csv, creates a linear model, requests user data, and generates a prediction.
   	* @param args: location of input file
	*/
	public static void main(String[] args) throws Exception {
		if(args.length != 1) {
			System.out.println("Please provide an input file for training");
			System.exit(2);
		}
		//capture and normalize data
		DataSource source = new DataSource(args[0]);
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes() - 1);
		Normalize filter = new Normalize();
		filter.setInputFormat(dataset);
		dataset = Filter.useFilter(dataset, filter);
		
		//create a linear regression model
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(dataset);

		Instance inst = new DenseInstance(11);

		Scanner s = new Scanner(System.in);

		//request data from user
		System.out.println("Please enter age:");
		double age = s.nextDouble();
		inst.setValue(new Attribute("age", 0), age);

		System.out.println("Please enter sex(male/female):");
		String sex = s.next();

		if (sex.equals("male")) {
			inst.setValue(new Attribute("sex", 1), (double) 1);
		} else if (sex.equals("female")) {
			inst.setValue(new Attribute("sex", 1), (double) -1);
		}

		System.out.println("Please enter bmi:");
		double bmi = s.nextDouble();
		inst.setValue(new Attribute("bmi", 2), bmi);

		System.out.println("Please enter number of children:");
		double children = s.nextDouble();
		inst.setValue(new Attribute("children", 3), children);

		System.out.println("Please enter smoker or not(1 = yes, 0 = no):");
		double smoker = s.nextDouble();
		inst.setValue(new Attribute("smoker", 4), smoker);

		System.out.println("Please enter location(southwest/southeast/northwest/northeast):");
		String loc = s.next();
		if (loc.equals("southwest")) {
			inst.setValue(new Attribute("southwest", 5), (double) 1);
			inst.setValue(new Attribute("southeast", 6), (double) 0);
			inst.setValue(new Attribute("northwest", 7), (double) 0);
			inst.setValue(new Attribute("northeast", 8), (double) 0);
		} else if (loc.equals("southeast")) {
			inst.setValue(new Attribute("southwest", 5), (double) 0);
			inst.setValue(new Attribute("southeast", 6), (double) 1);
			inst.setValue(new Attribute("northwest", 7), (double) 0);
			inst.setValue(new Attribute("northeast", 8), (double) 0);
		} else if (loc.equals("northwest")) {
			inst.setValue(new Attribute("southwest", 5), (double) 0);
			inst.setValue(new Attribute("southeast", 6), (double) 0);
			inst.setValue(new Attribute("northwest", 7), (double) 1);
			inst.setValue(new Attribute("northeast", 8), (double) 0);
		} else if (loc.equals("northeast")) {
			inst.setValue(new Attribute("southwest", 5), (double) 0);
			inst.setValue(new Attribute("southeast", 6), (double) 0);
			inst.setValue(new Attribute("northwest", 7), (double) 0);
			inst.setValue(new Attribute("northeast", 8), (double) 1);
		}

		inst.setDataset(dataset);
		
		//make prediction and print to the user
		Double pred = lr.classifyInstance(inst);
		System.out.println("Expected cost: $" + String.format("%.2f", pred));

	}

}
