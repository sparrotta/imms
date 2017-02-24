package it.uniroma1.lcl.imms.feature;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.util.CoreMap;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.Constants.HeadAnnotation;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;

public class WordEmbeddingFeaturizer implements Annotator {

	Map<String,double[]> wordMap = new HashMap<String, double[]>();
	protected int vectorSize = -1;
	public static final String DEFAULT_WINDOWSIZE = "10";
	Integer windowSize;

	public WordEmbeddingFeaturizer(Properties properties) {
		try {

			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(properties.getProperty(Constants.PROPERTY_IMMS_WORDEMBED_FILE))));

			String line;

			while ((line = reader.readLine()) != null) {
				try {
					final String actualLine = line;
					String array[] = actualLine.split(" ");
					double vector[] = new double[array.length - 1];
					for (int i = 0; i < vector.length; i++) {
						vector[i] = Double.parseDouble(array[i + 1]);
					}
					wordMap.put(array[0].trim(), vector);
				} catch (Exception e) {
					// corrupted line
					System.err.println("Corrupted line: " + line);
				}
			}
			vectorSize = wordMap.values().iterator().next().length;
			reader.close();

		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		windowSize = Integer.valueOf(properties.getProperty(Constants.PROPERTY_IMMS_WORDEMBED_WINDOWSIZE,DEFAULT_WINDOWSIZE));
	}

	@Override
	public void annotate(Annotation annotation) {
		CoreLabel head = annotation.get(HeadAnnotation.class);
		List<CoreLabel> tokens = null;
		for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
			if(sentence.get(CoreAnnotations.CharacterOffsetBeginAnnotation.class)>head.endPosition() || sentence.get(CoreAnnotations.CharacterOffsetEndAnnotation.class)<head.beginPosition()){
				continue;
			} else {
				tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
			}
		}

		List<Feature> features = new ArrayList<Feature>();

		if(tokens!=null){
			Integer headIndex=null;
			for(int i=0; i< tokens.size(); i++){
				if(tokens.get(i).beginPosition()==head.beginPosition()){
					headIndex=i;
					break;
				}
			}
			if(headIndex!=null){
				for(int i=0, slot=0; slot < windowSize*2; i++){
					if(i==headIndex){
						continue;
					} else {
						for(int j=0, pos=slot*vectorSize; j<vectorSize; pos++,j++){
							double value = 0.0;
							if (i<tokens.size()){
								value = wordMap.get(tokens.get(i).lemma().toLowerCase())[j];
							}
							features.add(new Feature<Double>("WE_"+pos,value));
						}
					}

					slot++;
				}
			}
		}

		head.get(Constants.FeaturesAnnotation.class).addAll(features);
		

	}

	@Override
	public Set<Requirement> requirementsSatisfied() {
		return Collections.singleton(Constants.IMMS_WORDEMBED_REQUIREMENT);
	}

	@Override
	public Set<Requirement> requires() {
		return TOKENIZE_SSPLIT_POS_LEMMA;
	}

}
