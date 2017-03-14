package it.uniroma1.lcl.imms.annotator;

import java.util.Properties;

import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.pipeline.AnnotatorImplementations;
import it.uniroma1.lcl.imms.annotator.feature.LocalCollocationFeatureAnnotator;
import it.uniroma1.lcl.imms.annotator.feature.POSFeatureAnnotator;
import it.uniroma1.lcl.imms.annotator.feature.SorroundingWordsFeatureAnnotator;
import it.uniroma1.lcl.imms.annotator.feature.WordEmbeddingFeatureAnnotator;

public class IMMSAnnotatorImplementations extends AnnotatorImplementations {

	public Annotator sorroundingWords(Properties properties){
		return new SorroundingWordsFeatureAnnotator(properties);
	}

	public Annotator sorroundingPOS(Properties inputProps) {
		return new POSFeatureAnnotator(inputProps);
	}

	public Annotator localCollocations(Properties inputProps) {
		return new LocalCollocationFeatureAnnotator(inputProps);
		
	}

	public Annotator wordEmbeddings(Properties inputProps) {
		return new WordEmbeddingFeatureAnnotator(inputProps);
	}
	
	public Annotator openNLPPosTagger(Properties properties) {		
		return new OpenNlpPosTaggerAnnotator(properties);		
	}

	public Annotator openNLPTokenizer(Properties properties) {		
		return new OpenNlpTokenizeAnnotator(properties);
	}

	public Annotator wordnetLemmatizer(Properties properties) {		
		return new WordNetLemmaAnnotator(properties);
	}
	
	public Annotator headToken(Properties properties) {		
		return new HeadTokenAnnotator(properties);
	}
}
