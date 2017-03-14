package it.uniroma1.lcl.imms.classifiers;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;

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
	static final String DEFAULT_SOLVER = SolverType.L2R_L2LOSS_SVC_DUAL.name();
	Parameter parameter;
	private double bias;
	
	public LibLinearClassifier(Properties properties) {
		super(properties);
		bias = Double.parseDouble(getProperties().getProperty(Constants.PROPERTY_IMMS_LIBLINEAR_BIAS, DEFAULT_BIAS));
		SolverType solver = SolverType.valueOf(getProperties().getProperty(Constants.PROPERTY_IMMS_LIBLINEAR_SOLVER, DEFAULT_SOLVER));
		String eps = getProperties().getProperty(Constants.PROPERTY_IMMS_LIBLINEAR_EPS);
		parameter = new Parameter(solver, 1, eps!=null ? Double.parseDouble(eps) : Double.POSITIVE_INFINITY);
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
		prob.n = d.featureIndex.size() + (prob.bias >= 0 ? 1 : 0);
		prob.x = new Feature[prob.l][];
		prob.y = new double[prob.l];
		int[] labels = d.getLabelsArray();
		for (int i = 0; i < prob.l; i++) {
			RVFDatum<String, String> datum = d.getRVFDatum(i);
			Collection<Feature> featureNodes = asFeatureNodes(datum.asFeaturesCounter(), d.featureIndex,bias);			
			prob.x[i] = featureNodes.toArray(new Feature[featureNodes.size()]);
			prob.y[i] = labels[i];
		}
		
		return prob;
	}	
	
	Collection<Feature> asFeatureNodes(Counter<String> featuresCounter,Index featureIndex,double bias){
		List<String> featureNames = new ArrayList<String>(featuresCounter.keySet());
		featureNames.sort(new Comparator<String>(){
			@Override
			public int compare(String o1, String o2) {
				int i1 = featureIndex.indexOf(o1);
				int i2 = featureIndex.indexOf(o2);
				if(i1==i2){ 
					return 0;
				} else {
					return i1<i2 ? -1 : 1;
				}				
			}			
		});
		List<Feature> featureNodes = new ArrayList<Feature>();
		for(String feature : featureNames){
			int fID = featureIndex.indexOf(feature);				
			if(fID>-1){
				featureNodes.add(new FeatureNode(fID+1, featuresCounter.getCount(feature)));
			}
		}		
		if (bias >= 0) {
			featureNodes.add(new FeatureNode(featureIndex.size(), bias));
		}
		return featureNodes;
	}
	
	

	@Override
	public List<String> test(String lexElem) {
		RVFDataset<String,String> d = dataset(lexElem);
		int[] labels = d.getLabelsArray();
		List<String> answers = new ArrayList<String>();
				
		for (int i = 0; i < d.size(); i++) {			
						
			Model model = model(lexElem);			
			Collection<Feature> featureNodes = asFeatureNodes(d.getRVFDatum(i).asFeaturesCounter(), d.featureIndex, model.getBias());
			Feature[] instance =  featureNodes.toArray(new Feature[featureNodes.size()]);
			
			double answer = Linear.predict(model, instance);			
			
			labels[i]=(new Double(answer).intValue());			
			answers.add(d.labelIndex.get(new Double(answer).intValue()));			
		}
		return answers;
	}
}
