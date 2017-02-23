package it.uniroma1.lcl.imms;

import java.util.Properties;

import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.pipeline.AnnotatorFactory;
import edu.stanford.nlp.pipeline.AnnotatorImplementations;
import edu.stanford.nlp.pipeline.AnnotatorPool;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import it.uniroma1.lcl.imms.feature.LocalCollocationFeaturizer;
import it.uniroma1.lcl.imms.feature.POSFeaturizer;
import it.uniroma1.lcl.imms.feature.SorroundingWordsFeaturizer;

public class IMMSAnnotationPipeline extends StanfordCoreNLP {

	public IMMSAnnotationPipeline(Properties props) {
		super(props);
	}

	@Override
	protected synchronized AnnotatorPool getDefaultAnnotatorPool(Properties inputProps,
			AnnotatorImplementations annotatorImplementation) {
		AnnotatorPool superPool = super.getDefaultAnnotatorPool(inputProps, annotatorImplementation);
		superPool.register(Constants.IMMS_SRNDWORDS, new AnnotatorFactory(inputProps, annotatorImplementation) {

			@Override
			public Annotator create() {
				return new SorroundingWordsFeaturizer(inputProps);
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
		});
		superPool.register(Constants.IMMS_POS, new AnnotatorFactory(inputProps, annotatorImplementation) {

			@Override
			public Annotator create() {
				return new POSFeaturizer(inputProps);
			}

			@Override
			protected String additionalSignature() {
				return Constants.PROPERTY_IMMS_POS_WINDOWSIZE+"="+inputProps.getProperty(Constants.PROPERTY_IMMS_POS_WINDOWSIZE, POSFeaturizer.DEFAULT_WINDOWSIZE);
			}
		});
		superPool.register(Constants.IMMS_LCOLLOCATION, new AnnotatorFactory(inputProps, annotatorImplementation) {

			@Override
			public Annotator create() {
				return new LocalCollocationFeaturizer(inputProps);
			}

			@Override
			protected String additionalSignature() {
				return Constants.PROPERTY_IMMS_LCOLLOCATIONSET+"="+inputProps.getProperty(Constants.PROPERTY_IMMS_LCOLLOCATIONSET, LocalCollocationFeaturizer.DEFAULT_COLLOCATIONS);
			}
		});
		return superPool;
	}

	// @Override
	// protected AnnotatorImplementations getAnnotatorImplementations() {
	//
	// String annotatorImplementationsClass =
	// getProperties().getProperty(Constants.CUSTOM_ANNOTATOR_IMPLEMENTATION_CLASS);
	// if(annotatorImplementationsClass != null){
	// try {
	// return (AnnotatorImplementations)
	// Class.forName(annotatorImplementationsClass).newInstance();
	// } catch (IllegalAccessException | InstantiationException |
	// ClassNotFoundException e) {
	// throw new RuntimeException(e);
	// }
	// } else {
	// return super.getAnnotatorImplementations();
	// }
	// }

}
