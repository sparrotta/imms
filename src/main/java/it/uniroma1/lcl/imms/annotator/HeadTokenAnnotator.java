package it.uniroma1.lcl.imms.annotator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.IDAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.util.CoreMap;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.Constants.HeadTokenAnnotation;
import it.uniroma1.lcl.imms.Constants.HeadsAnnotation;

public class HeadTokenAnnotator implements Annotator {

		


	public HeadTokenAnnotator(Properties properties) {
		
	}

	@Override
	public void annotate(Annotation annotation) {
		if (annotation.has(HeadsAnnotation.class) && annotation.has(TokensAnnotation.class)) {
		    List<CoreMap> heads = new ArrayList<CoreMap>();
		    heads.addAll(annotation.get(HeadsAnnotation.class));
			List<CoreLabel> tokens = annotation.get(TokensAnnotation.class);
			
			for(int i=0;i<tokens.size();i++){
				CoreLabel token = tokens.get(i);				
				CoreMap head=null;
				for(int j = 0; j < heads.size(); j++){
					if(token.beginPosition()==heads.get(j).get(CharacterOffsetBeginAnnotation.class)){
						head=heads.get(j);
						head.set(HeadTokenAnnotation.class, token);						
						break;						
					}					
				}
				if(head!=null){					
					heads.remove(head);
				}
				if(heads.isEmpty()){
					break;
				}
			}
			if(!heads.isEmpty()){				
				throw new RuntimeException("HeadTokenAnnotator unable to find tokens for heads: " + heads.stream()
					.map(head->head.get(IDAnnotation.class)+" "+head.get(CharacterOffsetBeginAnnotation.class)+":"+head.get(CharacterOffsetEndAnnotation.class)).collect(Collectors.toList())+
					" in annotation: "+annotation
				)
				;
			}
	    } else {
	      throw new RuntimeException("HeadTokensAnnotator unable to find heads or tokens in annotation: " + annotation);
	    }

	}

	@Override
	public Set<Requirement> requires() {
		return Collections.singleton(TOKENIZE_REQUIREMENT);
	}

	@Override
	public Set<Requirement> requirementsSatisfied() {
		return Collections.singleton(Constants.REQUIREMENT_ANNOTATOR_IMMS_HEADTOKEN);
	}

}
