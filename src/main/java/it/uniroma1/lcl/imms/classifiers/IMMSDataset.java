package it.uniroma1.lcl.imms.classifiers;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;
import it.uniroma1.lcl.imms.annotator.feature.Feature;

public class IMMSDataset {

	private double[][] values;
	private ArrayList<Pair<String, String>> sourcesAndIds;

	public Index<String> labelIndex;
	public Index<String> featureIndex;

	protected int[] labels;
	protected int[][] data;

	protected int size;

	public IMMSDataset() {
		this(10);
	}

	public IMMSDataset(int size) {
		labelIndex = new HashIndex<>();
		featureIndex = new HashIndex<>();
		labels = new int[size];
		data = new int[size][];
		values = new double[size][];
		sourcesAndIds = new ArrayList<>(size);
		size = 0;
	}

	private void addLabel(String label) {
		if (labels.length == size) {
			int[] newLabels = new int[size * 2];
			synchronized (System.class) {
				System.arraycopy(labels, 0, newLabels, 0, size);
			}
			labels = newLabels;
		}
		labels[size] = labelIndex.addToIndex(label);
	}

	private void addFeatures(Counter<String> features) {
		if (data.length == size) {
			int[][] newData = new int[size * 2][];
			double[][] newValues = new double[size * 2][];
			synchronized (System.class) {
				System.arraycopy(data, 0, newData, 0, size);
				System.arraycopy(values, 0, newValues, 0, size);
			}
			data = newData;
			values = newValues;
		}

		
		final List<String> featureNames = new ArrayList<String>();
		for(String feature : features.keySet()){
			if(featureIndex.addToIndex(feature)>-1){
				featureNames.add(feature);
			}
		}
		final int nFeatures = featureNames.size();
		data[size] = new int[nFeatures];
		values[size] = new double[nFeatures];
		for (int i = 0; i < nFeatures; ++i) {
			String feature = featureNames.get(i);
			data[size][i] = featureIndex.indexOf(feature);
			values[size][i] = features.getCount(feature);
		}
	}


	
	public RVFDatum<String, String> getRVFDatum(int index) {
		ClassicCounter<String> c = new ClassicCounter<>();
		for (int i = 0; i < data[index].length; i++) {
			c.setCount(featureIndex.get(data[index][i]), values[index][i]);
		}
		return new RVFDatum<>(c, labelIndex.get(labels[index]));
	}

	public String getRVFDatumSource(int index) {
		return sourcesAndIds.get(index).first();
	}

	public String getRVFDatumId(int index) {
		return sourcesAndIds.get(index).second();
	}

	public void add(RVFDatum<String,String> d, String src, String id) {
		addLabel(d.label());
		addFeatures(d.asFeaturesCounter());
		sourcesAndIds.add(new Pair<>(src, id));
		size++;
	}

	public int size() {
		
		return size;
	}
}
