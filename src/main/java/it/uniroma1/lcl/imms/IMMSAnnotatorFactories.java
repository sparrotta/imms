package it.uniroma1.lcl.imms;

import java.util.Properties;

import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.pipeline.AnnotatorFactory;
import it.uniroma1.lcl.imms.feature.LocalCollocationFeaturizer;
import it.uniroma1.lcl.imms.feature.POSFeaturizer;
import it.uniroma1.lcl.imms.feature.SorroundingWordsFeaturizer;
import it.uniroma1.lcl.imms.feature.WordEmbeddingFeaturizer;

public class IMMSAnnotatorFactories {

	public static AnnotatorFactory sorroundingWords(Properties inputProps,
			IMMSAnnotatorImplementations annotatorImplementation) {
		return new AnnotatorFactory(inputProps,annotatorImplementation) {
	
			private static final long serialVersionUID = 1L;

			@Override
			public Annotator create() {
				return annotatorImplementation.sorroundingWords(inputProps);
			}

			@Override
			protected String additionalSignature() {				
				StringBuilder sb = new StringBuilder();
				sb.append(Constants.PROPERTY_IMMS_SRNDWORDS_WINDOWSIZE+"=")
				.append(inputProps.getProperty(Constants.PROPERTY_IMMS_SRNDWORDS_WINDOWSIZE, SorroundingWordsFeaturizer.DEFAULT_WINDOWSIZE));
				
				if(inputProps.containsKey(Constants.PROPERTY_IMMS_SRNDWORDS_ADDSTOPWRD)){
					sb.append(Constants.PROPERTY_IMMS_SRNDWORDS_ADDSTOPWRD+"=")
					.append(inputProps.getProperty(Constants.PROPERTY_IMMS_SRNDWORDS_ADDSTOPWRD));
				}
				
				return sb.toString();
			}
		};
	}

	public static AnnotatorFactory sorroundingPOS(Properties inputProps,
			IMMSAnnotatorImplementations annotatorImplementation) {
		
		return new AnnotatorFactory(inputProps, annotatorImplementation) {

			private static final long serialVersionUID = 1L;

			@Override
			public Annotator create() {
				return annotatorImplementation.sorroundingPOS(inputProps);
			}

			@Override
			protected String additionalSignature() {
				return Constants.PROPERTY_IMMS_POS_WINDOWSIZE+"="+inputProps.getProperty(Constants.PROPERTY_IMMS_POS_WINDOWSIZE, POSFeaturizer.DEFAULT_WINDOWSIZE);
			}
		};
	}

	public static AnnotatorFactory localCollocation(Properties inputProps,
			IMMSAnnotatorImplementations annotatorImplementation) {
		
		return new AnnotatorFactory(inputProps, annotatorImplementation) {

			private static final long serialVersionUID = 1L;

			@Override
			public Annotator create() {
				return annotatorImplementation.localCollocations(inputProps);
			}

			@Override
			protected String additionalSignature() {
				return Constants.PROPERTY_IMMS_LCOLLOCATIONSET+"="+inputProps.getProperty(Constants.PROPERTY_IMMS_LCOLLOCATIONSET, LocalCollocationFeaturizer.DEFAULT_COLLOCATIONS);
			}
		};
	}

	public static AnnotatorFactory wordEmbeddings(Properties inputProps,
			IMMSAnnotatorImplementations annotatorImplementation) {

		return new AnnotatorFactory(inputProps, annotatorImplementation) {

			private static final long serialVersionUID = 1L;

			@Override
			public Annotator create() {
				return annotatorImplementation.wordEmbeddings(inputProps);
			}

			@Override
			protected String additionalSignature() {
				return Constants.PROPERTY_IMMS_WORDEMBED_WINDOWSIZE+"="+inputProps.getProperty(Constants.PROPERTY_IMMS_WORDEMBED_WINDOWSIZE, WordEmbeddingFeaturizer.DEFAULT_WINDOWSIZE);
			}
		};
	}

	
}
