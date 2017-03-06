package it.uniroma1.lcl.imms.task;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.pipeline.Annotation;

public interface ITaskHandler {

	void loadCorpus(String corpusFile) throws FileNotFoundException, IOException;
	void loadAnswers(String answersFile) throws FileNotFoundException, IOException;
	
	Iterator<Annotation> iterator();
	
	void writeResults(String property, String lexElem, RVFDataset<String,String> dataset);
	
	Map<String,Double> evaluate(Map<String,RVFDataset<String,String>> datasets);
}
