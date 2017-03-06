package it.uniroma1.lcl.imms.annotator.feature;

public class Feature<V> {

	private String key;
	private V value;
	public Feature(String key,V value) {
		this.key = key;
		this.value = value;
	}
	
	public String key() {return key; }
	public V value(){ return value; }
	
	@Override
	public int hashCode() {
		 int result = 17;
	        result = 31 * result + key.hashCode();
	        result = 31 * result + value.hashCode();
	        return result;
	}   
	
	@Override
	public boolean equals(Object o) {
		if (o == this) return true;
        if (!(o instanceof Feature)) {
            return false;
        }

        Feature that = (Feature<?>) o;

        return (that.key()==key  || (that.key()!=null && that.key().equals(key))) &&
                (that.value() == value || (that.value()!=null && that.value.equals(value)));
	}
	
	@Override
	public String toString() {	
		return "{"+key+": "+value+"}";
	}
}
