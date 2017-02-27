package it.uniroma1.lcl.imms;

import java.util.Properties;

import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.pipeline.AnnotatorImplementations;
import it.uniroma1.lcl.imms.feature.LocalCollocationFeaturizer;
import it.uniroma1.lcl.imms.feature.POSFeaturizer;
import it.uniroma1.lcl.imms.feature.SorroundingWordsFeaturizer;
import it.uniroma1.lcl.imms.feature.WordEmbeddingFeaturizer;

public class IMMSAnnotatorImplementations extends AnnotatorImplementations {

	public Annotator sorroundingWords(Properties properties){
		return new SorroundingWordsFeaturizer(properties);
	}

	public Annotator sorroundingPOS(Properties inputProps) {
		return new POSFeaturizer(inputProps);
	}

	public Annotator localCollocations(Properties inputProps) {
		return new LocalCollocationFeaturizer(inputProps);
		
	}

	public Annotator wordEmbeddings(Properties inputProps) {
		return new WordEmbeddingFeaturizer(inputProps);
	}
}
