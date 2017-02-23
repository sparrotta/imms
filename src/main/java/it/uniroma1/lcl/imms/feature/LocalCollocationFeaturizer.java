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
import it.uniroma1.lcl.imms.Constants.HeadAnnotation;

public class LocalCollocationFeaturizer implements Annotator {

	public static final String DEFAULT_COLLOCATIONS = "-2:-2,-1:-1,1:1,2:2,-2:-1,-1:1,1:2,-3:-1,-2:1,-1:2,1:3";
	String collocations;
	
	public LocalCollocationFeaturizer(Properties properties) {
		collocations = properties.getProperty(Constants.PROPERTY_IMMS_LCOLLOCATIONSET, DEFAULT_COLLOCATIONS);
	}

	@Override
	public void annotate(Annotation annotation) {
		CoreLabel head = annotation.get(HeadAnnotation.class);
		List<CoreLabel> tokens = null;				 		
		List<Feature> features = new ArrayList<Feature>();				
		List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
		for (CoreMap sentence : sentences) {
			if(sentence.get(CharacterOffsetBeginAnnotation.class)>head.endPosition() || sentence.get(CharacterOffsetEndAnnotation.class)<head.beginPosition()){
				continue;
			} else {
				tokens = sentence.get(TokensAnnotation.class);
			}			
		}
		if(tokens!=null){			
			Integer headIndex=null;
			for(int i=0; i< tokens.size(); i++){				
				if(tokens.get(i).beginPosition()==head.beginPosition()){
					headIndex=i;
					break;
				}
			}
			if(headIndex!=null){
				for(String collIndexPair:collocations.split(",")){
					String [] pair = collIndexPair.split(":");
					int start = Integer.parseInt(pair[0]);
					int end = Integer.parseInt(pair[1]);
					String key = "LC"+start+":"+end;
					for(int i=headIndex+start;i<=headIndex+end;i++){
						if(i==headIndex){
							continue;
						} else if(i>-1 && i<tokens.size()){
							key+="_"+tokens.get(i).lemma();
						} else {
							key+="_^";
						}
					}
					features.add(new Feature<Boolean>(key,true));
				}
			}
		}
		head.get(Constants.FeaturesAnnotation.class).addAll(features);
		

	}

	@Override
	public Set<Requirement> requirementsSatisfied() {
		return Collections.singleton(Constants.IMMS_LCOLLOCATION_REQUIREMENT);
	}

	@Override
	public Set<Requirement> requires() {
		return TOKENIZE_SSPLIT_POS_LEMMA;
	}

}
