package it.uniroma1.lcl.imms.annotator.feature;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.util.ArraySet;
import edu.stanford.nlp.util.CoreMap;
import it.uniroma1.lcl.imms.Constants;
import it.uniroma1.lcl.imms.Constants.HeadTokenAnnotation;
import it.uniroma1.lcl.imms.Constants.HeadsAnnotation;

public class WordEmbeddingFeatureAnnotator implements Annotator {

	public enum Strategies {
		concatenation,
		fractional,
		average,
		exponential
	}
	
	public static final String DEFAULT_WINDOWSIZE = "10";
	public static final String DEFAULT_STRATEGY = "exponential";
	
	Map<String,double[]> wordMap = new HashMap<String, double[]>();
	protected int vectorSize = -1;
	
	Integer windowSize;
	private Strategies strategy;
	private double decay;

	public WordEmbeddingFeatureAnnotator(Properties properties) {
		BufferedReader reader = null;
		try {

			String filename = properties.getProperty(Constants.PROPERTY_IMMS_WORDEMBED_FILE);
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
			System.out.print("\tdone.\n");
			
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
		windowSize = Integer.valueOf(properties.getProperty(Constants.PROPERTY_IMMS_WORDEMBED_WINDOWSIZE,DEFAULT_WINDOWSIZE));
		decay = 1 - Math.pow(0.1,1/windowSize-1);
		strategy = Strategies.valueOf(Strategies.class,properties.getProperty(Constants.PROPERTY_IMMS_WORDEMBED_STRATEGY,DEFAULT_STRATEGY));
	}

	@Override
	public void annotate(Annotation annotation) {
		for(CoreMap head : annotation.get(HeadsAnnotation.class)){
			head.get(Constants.FeaturesAnnotation.class).addAll(featurize(head.get(HeadTokenAnnotation.class), annotation));
		}
	}

	private List<Feature> featurize(CoreLabel head, Annotation annotation) {
		List<Feature> features=null;
		switch(strategy){		
		case concatenation: features = concatenation(head, annotation);break;
		case exponential: features = exponential(head, annotation);break;
		case fractional: features = fractional(head, annotation);
		case average: 
		default: features = average(head, annotation);
		}
		return features;			
	}
	
	private List<Feature> fractional(CoreLabel head, Annotation annotation) {
		List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
		List<Feature> features = new ArrayList<Feature>();
		Integer headIndex=head.index();
		
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
			features.add(new Feature<Double>("WE_"+i,v[i]/windowSize));
		}
		
		return features;

	}
	
	private List<Feature> exponential(CoreLabel head, Annotation annotation) {
		List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
		List<Feature> features = new ArrayList<Feature>();
		Integer headIndex=head.index();
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
			features.add(new Feature<Double>("WE_"+i,v[i]));
		}
		return features;

	}
	
	private List<Feature> average(CoreLabel head, Annotation annotation) {
		List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
		List<Feature> features = new ArrayList<Feature>();
		Integer headIndex=head.index();
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
			features.add(new Feature<Double>("WE_"+i,v[i]/(2*windowSize)));
		}
		return features;

	}
	
	private List<Feature> concatenation(CoreLabel head, Annotation annotation) {
		List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
		List<Feature> features = new ArrayList<Feature>();
		
		Integer headIndex=head.index();
		for(int i=headIndex-windowSize, slot=0; i<= headIndex+windowSize; i++){
			if(i==headIndex){
				continue;
			} else {
				for(int j=0, pos=slot*vectorSize; j<vectorSize; pos++,j++){
					double value = 0.0;
					if (i>-1 && i<tokens.size()){
						String word = tokens.get(i).lemma().toLowerCase();
						if(wordMap.containsKey(word)){
							value = wordMap.get(word)[j];
						}								
					}						
					features.add(new Feature<Double>("WE_"+pos,value));
				}
			}

			slot++;
		}

		return features;			
	}

	@Override
	public Set<Requirement> requirementsSatisfied() {
		return Collections.singleton(Constants.REQUIREMENT_ANNOTATOR_FEAT_IMMS_WORDEMBED);
	}

	@Override
	public Set<Requirement> requires() {
		return Collections.unmodifiableSet(new ArraySet<>(TOKENIZE_REQUIREMENT, SSPLIT_REQUIREMENT,LEMMA_REQUIREMENT,Constants.REQUIREMENT_ANNOTATOR_IMMS_HEADTOKEN));
	}

}
