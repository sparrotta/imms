package it.uniroma1.lcl.imms.classifiers;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Properties;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ling.Datum;

public class LibLinearClassifier extends Classifier<Model> {
	
	public LibLinearClassifier(Properties properties) {
		super(properties);
	}

	@Override
	public Model train(String lexElem) {		
		Problem prob = getProblem(lexElem);
		return Linear.train(prob, new Parameter(SolverType.L2R_L2LOSS_SVC_DUAL, 1, Double.POSITIVE_INFINITY));		
	}

	private Problem getProblem(String lexElement) {		
		Problem prob = new Problem();
		Dataset<String,it.uniroma1.lcl.imms.feature.Feature<?>>d = dataset(lexElement);
		// TODO allow bias: if >0 add the bias feature
		prob.bias = Double.parseDouble(getProperties().getProperty("bias", "-1.0"));	
		prob.l = d.size();
		prob.n = d.numFeatures() + (prob.bias > 0 ? 1 : 0);
		prob.x = new Feature[prob.l][];
		prob.y = new double[prob.l];

		for (int i = 0; i < prob.l; i++) {
			Datum<String, it.uniroma1.lcl.imms.feature.Feature<?>> datum = d.getDatum(i);
			List<Feature> featureNodes = new ArrayList<Feature>();
			for (it.uniroma1.lcl.imms.feature.Feature<?> feature : datum.asFeatures()) {
				featureNodes.add(new FeatureNode(d.featureIndex().indexOf(feature) + 1, toDouble(feature.value())));
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
			if (prob.bias > 0) {
				featureNodes.add(new FeatureNode(prob.n, 1.0));
			}
			prob.x[i] = featureNodes.toArray(new Feature[featureNodes.size()]);
			prob.y[i] = d.labelIndex().indexOf(datum.label());
		}
		
		return prob;
	}

	Double toDouble(Object o) {
		Double d = 0.0;
		if (o instanceof Boolean) {
			d = ((Boolean) o) ? 1.0 : 0.0;
		} else if (o instanceof Number) {
			d = ((Number) o).doubleValue();
		} else if (o instanceof String) {
			d = Double.valueOf((String) o);
		}
		return d;
	}

	@Override
	public List<String> test(String lexElem) {
		Dataset<String,it.uniroma1.lcl.imms.feature.Feature<?>> d = dataset(lexElem);		
		List<String> answers = new ArrayList<String>();
		Problem prob = getProblem(lexElem);		
		for (int i = 0; i < prob.l; i++) {
			Feature[] instance = prob.x[i];											
			double answer = Linear.predict(model(lexElem), instance);
			answers.add(d.labelIndex.get(new Double(answer).intValue()));			
		}
		return answers;
	}
}
