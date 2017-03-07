package it.uniroma1.lcl.imms.classifiers;

import de.bwaldvogel.liblinear.*;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Index;
import it.uniroma1.lcl.imms.Constants;

import java.util.*;
import java.util.Map.Entry;

public class LibLinearClassifier extends Classifier<Model> {
	
	static final String DEFAULT_BIAS = "-1.0";
	static final String DEFAULT_SOLVER = SolverType.L2R_L2LOSS_SVC_DUAL.name();

	Parameter parameter;
	private double bias;
	
	public LibLinearClassifier(Properties properties) {
		super(properties);
		bias = Double.parseDouble(getProperties().getProperty(Constants.PROPERTY_IMMS_LIBLINEAR_BIAS, DEFAULT_BIAS));
		SolverType type = SolverType.valueOf((getProperties().getProperty(Constants.PROPERTY_IMMS_LIBLINEAR_SOLVER, DEFAULT_SOLVER)));
		String eps = getProperties().getProperty(Constants.PROPERTY_IMMS_LIBLINEAR_SOLVER_EPS);
		parameter = new Parameter(type, 1, eps!=null ? Double.parseDouble(eps) : Double.POSITIVE_INFINITY);
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
		IMMSDataset d = dataset(lexElement);
		prob.bias = bias;	
		prob.l = d.size;
		prob.n = d.featureIndex.size() + (prob.bias >= 0 ? 1 : 0);
		prob.x = new Feature[prob.l][];
		prob.y = new double[prob.l];

		for (int i = 0; i < prob.l; i++) {
			RVFDatum<String, String> datum = d.getRVFDatum(i);
			Collection<Feature> featureNodes = asFeatureNodes(datum.asFeaturesCounter(), d.featureIndex);			
			prob.x[i] = featureNodes.toArray(new Feature[featureNodes.size()]);
			prob.y[i] = d.labelIndex.indexOf(datum.label());
		}
		
		return prob;
	}	
	
	Collection<Feature> asFeatureNodes(Counter<String> featuresCounter,Index featureIndex){
		List<Feature> featureNodes = new ArrayList<Feature>();
		for (Entry<String,Double> feature : featuresCounter.entrySet()) {
			int fID = featureIndex.indexOf(feature.getKey());
			if(fID>-1){
				featureNodes.add(new FeatureNode(fID+1, feature.getValue()));
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
		IMMSDataset d = dataset(lexElem);
		
		List<String> answers = new ArrayList<String>();
				
		for (int i = 0; i < d.size; i++) {			
			Collection<Feature> featureNodes = asFeatureNodes(d.getRVFDatum(i).asFeaturesCounter(), d.featureIndex);			
			Feature[] instance =  featureNodes.toArray(new Feature[featureNodes.size()]);														
			double answer = Linear.predict(model(lexElem), instance);			
			
			d.labels[i]=(new Double(answer).intValue());			
			answers.add(d.labelIndex.get(new Double(answer).intValue()));			
		}
		return answers;
	}
}
