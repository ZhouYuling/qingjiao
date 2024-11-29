package mobile;

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Objects;

public class MobileBean implements Writable {

    private Double price;
    private Integer sales;
    private Integer rate;

    public MobileBean() {
    }

    public MobileBean(Double price, Integer sales, Integer rate) {
        this.price = price;
        this.sales = sales;
        this.rate = rate;
    }

    public Double getPrice() {
        return price;
    }

    public void setPrice(Double price) {
        this.price = price;
    }

    public Integer getSales() {
        return sales;
    }

    public void setSales(Integer sales) {
        this.sales = sales;
    }

    public Integer getRate() {
        return rate;
    }

    public void setRate(Integer rate) {
        this.rate = rate;
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) return false;
        MobileBean that = (MobileBean) o;
        return Objects.equals(price, that.price) && Objects.equals(sales, that.sales) && Objects.equals(rate, that.rate);
    }

    @Override
    public int hashCode() {
        return Objects.hash(price, sales, rate);
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        dataOutput.writeDouble(price);
        dataOutput.writeInt(sales);
        dataOutput.writeInt(rate);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        this.price = dataInput.readDouble();
        this.sales = dataInput.readInt();
        this.rate = dataInput.readInt();
    }

    @Override
    public String toString() {
        return price + "\t" + sales + "\t" + rate;
    }
}
