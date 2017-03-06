package it.uniroma1.lcl.imms.annotator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.pipeline.Annotator.Requirement;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.Span;

public class OpenNlpTokenizeAnnotator implements Annotator {

	public static final String ANNOTATION_NAME = "otokenize";
	public static final Requirement OTOKENIZE_REQUIREMENT = new Requirement(ANNOTATION_NAME);		
	public static final String PROPERTY_MODEL = ANNOTATION_NAME+".model";
	private TokenizerME tokenizer;
	
	public OpenNlpTokenizeAnnotator(Properties props) {
		String modelPath = props.getProperty(PROPERTY_MODEL);		
		if (modelPath != null) {
			try {
				this.tokenizer = new TokenizerME(new TokenizerModel(new File(modelPath)));
			} catch (IOException e) {
				throw new RuntimeException(e);
			}			
		}
	}

	@Override
	public void annotate(Annotation annotation) {
		if (annotation.has(CoreAnnotations.TextAnnotation.class)) {
		      String text = annotation.get(CoreAnnotations.TextAnnotation.class);
		      
		      // don't wrap in BufferedReader.  It gives you nothing for in-memory String unless you need the readLine() method!
		      List<CoreLabel> tokens = new ArrayList<CoreLabel>();
		      Span[] spans = tokenizer.tokenizePos(text);
		      for(int i=0; i<spans.length;i++){
		    	  
		    	  CoreLabel cl = new CoreLabel();
		    	  cl.setBeginPosition(spans[i].getStart());
		    	  cl.setEndPosition(spans[i].getEnd());
		    	  String tokenText = text.substring(cl.beginPosition(), cl.endPosition());
		    	  cl.setValue(tokenText);
		    	  cl.setWord(tokenText);
		    	  cl.setOriginalText(tokenText);
		    	  tokens.add(cl);
		      }
		       
		      // cdm 2010-05-15: This is now unnecessary, as it is done in CoreLabelTokenFactory
		      // for (CoreLabel token: tokens) {
		      // token.set(CoreAnnotations.TextAnnotation.class, token.get(CoreAnnotations.TextAnnotation.class));
		      // }
		      annotation.set(CoreAnnotations.TokensAnnotation.class, tokens);		      
		    } else {
		      throw new RuntimeException("Tokenizer unable to find text in annotation: " + annotation);
		    }

	}

	@Override
	public Set<Requirement> requires() {
		return Collections.emptySet();
	}

	@Override
	public Set<Requirement> requirementsSatisfied() {
		return Collections.unmodifiableSet(new HashSet<Requirement>() {{
	        add(OTOKENIZE_REQUIREMENT);
	        add(TOKENIZE_REQUIREMENT);
	      }});
	}

}
