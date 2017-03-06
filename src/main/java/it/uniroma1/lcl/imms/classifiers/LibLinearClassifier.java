package it.uniroma1.lcl.imms.classifiers;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Index;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.Constants.LexicalItemAnnotation;

public class LibLinearClassifier extends Classifier<Model> {
	
	static final String DEFAULT_BIAS = "-1.0";
	Parameter parameter = new Parameter(SolverType.L2R_L2LOSS_SVC_DUAL, 1, Double.POSITIVE_INFINITY);
	private double bias;
	
	public LibLinearClassifier(Properties properties) {
		super(properties);
		bias = Double.parseDouble(getProperties().getProperty(Constants.PROPERTY_IMMS_LIBLINEAR_BIAS, DEFAULT_BIAS));
	}

	@Override
	public Model train(String lexElem) {		
		Problem prob = getProblem(lexElem);
		if (this.parameter.getEps() == Double.POSITIVE_INFINITY) {
			if (this.parameter.getSolverType() == SolverType.L2R_LR || this.parameter.getSolverType() == SolverType.L2R_L2LOSS_SVC) {
				this.parameter.setEps(0.01);
			} else if (this.parameter.getSolverType() == SolverType.L2R_L2LOSS_SVC_DUAL || this.parameter.getSolverType() == SolverType.L2R_L1LOSS_SVC_DUAL
					|| this.parameter.getSolverType() == SolverType.MCSVM_CS) {
				this.parameter.setEps(0.1);
			}
		}
		return Linear.train(prob, this.parameter);		
	}

	private Problem getProblem(String lexElement) {		
		Problem prob = new Problem();
		RVFDataset<String,String> d = dataset(lexElement);
		prob.bias = bias;	
		prob.l = d.size();
		prob.n = d.numFeatures() + (prob.bias >= 0 ? 1 : 0);
		prob.x = new Feature[prob.l][];
		prob.y = new double[prob.l];

		for (int i = 0; i < prob.l; i++) {
			RVFDatum<String, String> datum = d.getRVFDatum(i);
			Collection<Feature> featureNodes = asFeatureNodes(datum.asFeaturesCounter(), d.featureIndex);			
			prob.x[i] = featureNodes.toArray(new Feature[featureNodes.size()]);
			prob.y[i] = d.labelIndex().indexOf(datum.label());
		}
		
		return prob;
	}	
	
	Collection<Feature> asFeatureNodes(Counter<String> featuresCounter,Index featureIndex){
		List<Feature> featureNodes = new ArrayList<Feature>();
		for (Entry<String,Double> feature : featuresCounter.entrySet()) {
			int featIndex = featureIndex.indexOf(feature.getKey());
			if(featIndex>-1){
				featureNodes.add(new FeatureNode(featIndex + 1, feature.getValue()));
			}				
		}
		featureNodes.sort(new Comparator<Feature>() {
			@Override
			public int compare(Feature o1, Feature o2) {
				if (o1.getIndex() == o2.getIndex()) {
					return 0;
				} else {
					return o1.getIndex() < o2.getIndex() ? -1 : 1;
				}
			}
		});
		if (bias >= 0) {
			featureNodes.add(new FeatureNode(featureIndex.size(), bias));
		}
		return featureNodes;
	}
	
	

	@Override
	public List<String> test(String lexElem) {
		RVFDataset<String,String> d = dataset(lexElem);
		
		List<String> answers = new ArrayList<String>();
				
		for (int i = 0; i < d.size(); i++) {			
			Collection<Feature> featureNodes = asFeatureNodes(d.getRVFDatum(i).asFeaturesCounter(), d.featureIndex);			
			Feature[] instance =  featureNodes.toArray(new Feature[featureNodes.size()]);														
			double answer = Linear.predict(model(lexElem), instance);			
			
			d.getLabelsArray()[i]=(new Double(answer).intValue());			
			answers.add(d.labelIndex.get(new Double(answer).intValue()));			
		}
		return answers;
	}
}
