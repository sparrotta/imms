package it.uniroma1.lcl.imms.annotator.feature;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.DoubleSummaryStatistics;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.SentenceIndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.pipeline.Annotator.Requirement;
import edu.stanford.nlp.util.ArraySet;
import edu.stanford.nlp.util.CoreMap;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.Constants.HeadTokenAnnotation;
import it.uniroma1.lcl.imms.Constants.HeadsAnnotation;
import it.uniroma1.lcl.imms.annotator.HeadTokenAnnotator;

public class WordEmbeddingFeatureAnnotator implements Annotator {

	
	public enum Strategies {
		concatenation,
		fractional,
		average,
		exponential
	}
	
	public static final String ANNOTATION_NAME = "feat_wordembed";
	public static final String FEATURE_PREFIX = "WRDMB_";
	public static final Requirement REQUIREMENT = new Requirement(ANNOTATION_NAME);

	public static final String PROPERTY_WINDOWSIZE = ANNOTATION_NAME+ ".windowsize";
	public static final String PROPERTY_FILE = ANNOTATION_NAME + ".file";
	public static final String PROPERTY_STRATEGY = ANNOTATION_NAME + ".strategy";
	public static final String PROPERTY_SENTENCEBOUND = ANNOTATION_NAME + ".sentencebound";
	public static final String PROPERTY_SIGMA = ANNOTATION_NAME + ".sigma";
	
	public static final String DEFAULT_WINDOWSIZE = "10";
	public static final String DEFAULT_STRATEGY = "concatenation";
	public static final String DEFAULT_SENTENCEBOUND = "false";
	public static final String DEFAULT_SIGMA = "0";
	
	Map<String,double[]> wordMap = new HashMap<String, double[]>();
	protected int vectorSize = -1;
	
	Integer windowSize;
	private Strategies strategy;
	private double decay;
	private double sigma;
	private boolean sentenceBound;

	public WordEmbeddingFeatureAnnotator(Properties properties) {
		windowSize = Integer.valueOf(properties.getProperty(PROPERTY_WINDOWSIZE,DEFAULT_WINDOWSIZE));
		decay = 1 - Math.pow(0.1,(windowSize-1)*-1);
		strategy = Strategies.valueOf(Strategies.class,properties.getProperty(PROPERTY_STRATEGY,DEFAULT_STRATEGY));
		sigma = Double.valueOf(properties.getProperty(PROPERTY_SIGMA,DEFAULT_SIGMA));
		sentenceBound = Boolean.valueOf(properties.getProperty(PROPERTY_SENTENCEBOUND,DEFAULT_SENTENCEBOUND));
		BufferedReader reader = null;
		try {

			String filename = properties.getProperty(PROPERTY_FILE);
			System.out.print("Reading word embeddings from "+filename+" ...");
			reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
			
			String line;
			int cnt=0;
			while ((line = reader.readLine()) != null) {
				++cnt;
				if(cnt%1000==0){
					System.out.print("\rReading word embeddings from "+filename+" ..."+cnt);
				}
				
				try {
					final String actualLine = line;
					String array[] = actualLine.split("\\s");
					double vector[] = new double[array.length - 1];
					for (int i = 0; i < vector.length; i++) {
						vector[i] = Double.parseDouble(array[i + 1]);						
					}
					wordMap.put(array[0].trim().toLowerCase(), vector);					
				} catch (Exception e) {
					// corrupted line
					System.err.println("Corrupted line: " + line);
				}
				
			}
			
			vectorSize = wordMap.values().iterator().next().length;
			System.out.print(" done. Vectors dimension: "+vectorSize+"\n");
			if(sigma>0){
				System.out.print("Scaling word embeddings vectors...");
				double [] mean = new double[vectorSize];
				double [] variance = new double[vectorSize];
				for(double[] wordVect : wordMap.values()){
					for(int i=0; i < wordVect.length; i++){
						mean[i]+=wordVect[i]/wordMap.size();
					}
				}			
				for(double[] wordVect : wordMap.values()){
					for(int i=0; i < wordVect.length; i++){
						variance[i]+=Math.pow(wordVect[i]-mean[i],2)/wordMap.size();
					}
				}			
				for(double[] wordVect : wordMap.values()){				
					for(int i=0; i < wordVect.length; i++){
						wordVect[i]=(sigma/Math.sqrt(variance[i]))*wordVect[i];					
					}				
				}
				System.out.print("done.\n");
			}
			
			
			
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		} finally {
			if(reader !=null){
				try {
					reader.close();
				} catch (IOException e) {					
					e.printStackTrace();
				}
			}
		}
		
		
	}

	@Override
	public void annotate(Annotation annotation) {
		for(CoreMap head : annotation.get(HeadsAnnotation.class)){
			head.get(Constants.FeaturesAnnotation.class).addAll(featurize(head.get(HeadTokenAnnotation.class), annotation));
		}
	}

	private List<Feature> featurize(CoreLabel head, Annotation annotation) {						
		List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
		List<CoreLabel> tokens = null;
		if(sentenceBound){
			for (CoreMap sentence : sentences) {
				if(sentence.get(SentenceIndexAnnotation.class)==head.sentIndex()){			
					tokens = sentence.get(TokensAnnotation.class);
					break;
				}								
			}
		} else {
			tokens = annotation.get(TokensAnnotation.class);
		}
		
		List<Feature> features=null;
		if(tokens!=null){
			switch(strategy){		
			case concatenation: features = concatenation(head, tokens);break;
			case exponential: features = exponential(head, tokens);break;
			case fractional: features = fractional(head, tokens);
			case average: 
			default: features = average(head, tokens);
			}
		}
		return features;			
	}
	
	private List<Feature> fractional(CoreLabel head, List<CoreLabel> tokens) {
		
		List<Feature> features = new ArrayList<Feature>();
		Integer headIndex=head.index()-tokens.get(0).index();
		
		double[] v = new double[vectorSize];
		for(int i=headIndex-windowSize; i<= headIndex+windowSize; i++){
			if(i==headIndex){
				continue;
			} else if (i>-1 && i<tokens.size()){
				String word = tokens.get(i).lemma().toLowerCase();
				if(wordMap.containsKey(word)){							
					for(int j=0; j<vectorSize; j++){								
						v[j]+=wordMap.get(word)[j]*(windowSize-Math.abs(headIndex-i));						
					}
				}								
			}
		}
		for(int i=0; i < vectorSize; i++){
			features.add(new Feature<Double>(FEATURE_PREFIX+i,v[i]/windowSize));
		}
		
		return features;

	}
	
	private List<Feature> exponential(CoreLabel head, List<CoreLabel> tokens) {
		List<Feature> features = new ArrayList<Feature>();
		Integer headIndex=head.index()-tokens.get(0).index();
		double[] v = new double[vectorSize];
		for(int i=headIndex-windowSize; i<= headIndex+windowSize; i++){
			if(i==headIndex){
				continue;
			} else if (i>-1 && i<tokens.size()){
				String word = tokens.get(i).lemma().toLowerCase();
				if(wordMap.containsKey(word)){							
					for(int j=0; j<vectorSize; j++){								
						v[j]+=wordMap.get(word)[j]*Math.pow(1-decay,Math.abs(headIndex-i)-1);						
					}
				}																	
			}
		}
		for(int i=0; i < vectorSize; i++){
			features.add(new Feature<Double>(FEATURE_PREFIX+i,v[i]));
		}
		return features;

	}
	
	private List<Feature> average(CoreLabel head, List<CoreLabel> tokens) {
		
		List<Feature> features = new ArrayList<Feature>();
		Integer headIndex=head.index() - tokens.get(0).index();
		double[] v = new double[vectorSize];
		for(int i=headIndex-windowSize; i<= headIndex+windowSize; i++){
			if(i==headIndex){
				continue;
			} else if (i>-1 && i<tokens.size()){
				String word = tokens.get(i).lemma().toLowerCase();
				if(wordMap.containsKey(word)){							
					for(int j=0; j<vectorSize; j++){								
						v[j]+=wordMap.get(word)[j];						
					}
				}																	
			}
		}
		for(int i=0; i < vectorSize; i++){
			features.add(new Feature<Double>(FEATURE_PREFIX+i,v[i]/(2*windowSize)));
		}
		return features;

	}
	
	private List<Feature> concatenation(CoreLabel head, List<CoreLabel> tokens) {		
		List<Feature> features = new ArrayList<Feature>();
		
		Integer headIndex=head.index()-tokens.get(0).index();
		for(int i=headIndex-windowSize, slot=0; i<= headIndex+windowSize; i++){
			if(i==headIndex){
				continue;
			} else {
				for(int j=0, pos=slot*vectorSize; j<vectorSize; pos++,j++){
					double value = 0.0;
					if (i>-1 && i<tokens.size()){
						String word = tokens.get(i).lemma();
						if(wordMap.containsKey(word)){
							value = wordMap.get(word)[j];
						}								
					}						
					features.add(new Feature<Double>(FEATURE_PREFIX+pos,value));
				}
			}

			slot++;
		}

		return features;			
	}

	@Override
	public Set<Requirement> requirementsSatisfied() {
		return Collections.singleton(REQUIREMENT);
	}

	@Override
	public Set<Requirement> requires() {
		return Collections.unmodifiableSet(new ArraySet<>(TOKENIZE_REQUIREMENT, SSPLIT_REQUIREMENT,LEMMA_REQUIREMENT,HeadTokenAnnotator.REQUIREMENT));
	}

}
