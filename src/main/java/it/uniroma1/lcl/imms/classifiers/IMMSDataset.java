package it.uniroma1.lcl.imms.classifiers;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;
import it.uniroma1.lcl.imms.annotator.feature.Feature;

public class IMMSDataset {

	private double[][] values;
	private ArrayList<Pair<String, String>> sourcesAndIds;

	public Index<String> labelIndex;
	public Index<String> featureIndex;

	Map<Integer, Index<String>> featureValuesMap;
	protected int[] labels;
	protected int[][] data;

	protected int size;

	public IMMSDataset() {
		this(10);
	}

	public IMMSDataset(int size) {
		labelIndex = new HashIndex<>();
		featureIndex = new HashIndex<>();
		featureValuesMap = new HashMap<Integer, Index<String>>();
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

	private void addFeatures(Collection<Feature> features) {
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

		final int nFeatures = features.size();
		Feature[] featureArray = features.toArray(new Feature[nFeatures]);

		data[size] = new int[nFeatures];
		values[size] = new double[nFeatures];
		for (int i = 0; i < nFeatures; ++i) {
			Feature feature = featureArray[i];
			int fID = featureIndex.addToIndex(feature.key());
			if (fID >= 0) {
				data[size][i] = fID;
				values[size][i] = getFeatureValue(feature);
			} else {
				// Usually a feature present at test but not training time.
				assert featureIndex.isLocked() : "Could not add feature to index: " + feature;
			}
		}
	}

	double getFeatureValue(Feature feature) {
		Object featureValue = feature.value();
		if (featureValue instanceof Number) {
			return ((Number) featureValue).doubleValue();
		}
		int fID = featureIndex.indexOf(feature.key());
		if (fID > 0) {
			Index<String> featureValuesIndex = featureValuesMap.get(fID);
			if (featureValuesIndex == null) {
				featureValuesIndex = new HashIndex<>();
				featureValuesMap.put(fID, featureValuesIndex);
			}
			return 1+featureValuesIndex.addToIndex(featureValue.toString());
		}
		return 0;
	}

	public RVFDatum<String, String> getRVFDatum(int index) {
		ClassicCounter<String> c = new ClassicCounter<>();
		for (int i = 0; i < data[index].length; i++) {
			c.incrementCount(featureIndex.get(data[index][i]), values[index][i]);
		}
		return new RVFDatum<>(c, labelIndex.get(labels[index]));
	}

	public String getRVFDatumSource(int index) {
		return sourcesAndIds.get(index).first();
	}

	public String getRVFDatumId(int index) {
		return sourcesAndIds.get(index).second();
	}

	public int size(){
		return size;
	}
	public void add(Collection<Feature> features, String label, String src, String id) {
		addLabel(label);
		addFeatures(features);
		sourcesAndIds.add(new Pair<>(src, id));
		size++;
	}
}
