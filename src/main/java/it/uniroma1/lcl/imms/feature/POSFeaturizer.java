package it.uniroma1.lcl.imms.feature;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.util.CoreMap;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.Constants.HeadsAnnotation;

public class POSFeaturizer implements Annotator {

	public static final String DEFAULT_WINDOWSIZE = "3";
	
	Integer windowSize;
	
	public POSFeaturizer(Properties properties) {
		windowSize = Integer.valueOf(properties.getProperty(Constants.PROPERTY_IMMS_POS_WINDOWSIZE,DEFAULT_WINDOWSIZE));
	}

	@Override
	public void annotate(Annotation annotation) {
		for(CoreLabel head : annotation.get(HeadsAnnotation.class)){
			featurize(head, annotation);
		}
	}

	private void featurize(CoreLabel head, Annotation annotation) {
		List<String> before = new ArrayList<String>();
		List<String> after = new ArrayList<String>();
		List<Feature> features = new ArrayList<Feature>();		
		
		List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
		for (CoreMap sentence : sentences) {
			if(sentence.get(CharacterOffsetBeginAnnotation.class)>head.endPosition() || sentence.get(CharacterOffsetEndAnnotation.class)<head.beginPosition()){
				continue;
			}
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				if (token.endPosition() < head.beginPosition() && Constants.PREDICATE_IS_WORD.test(token.lemma())) {					
					before.add(token.tag());									
				} else if (token.beginPosition() > head.endPosition() && Constants.PREDICATE_IS_WORD.test(token.lemma())) {
					after.add(token.tag());					
				} else if(token.beginPosition()==head.beginPosition()){//it's the head token
					features.add(new Feature<Boolean>("P0_"+token.tag(),true));
				}				
			}
		}
		
		
		int beforeSize = before.size();
		for(int i=Math.max(0,beforeSize-windowSize); i<beforeSize; i++){			
			features.add(new Feature<Boolean>("P" + (i-beforeSize) + "_" + before.get(i), true));
		}
		int afterSize = after.size();
		for(int i=0;i<Math.min(windowSize,afterSize);i++){
			features.add(new Feature<Boolean>("P" + (i+1) + "_" + after.get(i), true));
		}
		head.get(Constants.FeaturesAnnotation.class).addAll(features);	
		
	}

	Feature posFeature(String tag, Integer position){		
		return new Feature<Boolean>("P" + position + "_" + tag, true);
	}
	
	@Override
	public Set<Requirement> requirementsSatisfied() {
		return Collections.singleton(Constants.IMMS_POS_REQUIREMENT);
	}

	@Override
	public Set<Requirement> requires() {
		return TOKENIZE_SSPLIT_POS_LEMMA;
	}

}
