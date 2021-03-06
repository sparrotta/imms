package it.uniroma1.lcl.imms.task;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.pipeline.Annotation;
import it.uniroma1.lcl.imms.classifiers.IMMSDataset;

public interface ITaskHandler {

	void loadCorpus(String corpusFile) throws FileNotFoundException, IOException;
	void loadAnswers(String answersFile) throws FileNotFoundException, IOException;
	
	Iterator<Annotation> iterator();
	
	void writeResults(String lexElem, RVFDataset<String,String> dataset);
	
}
