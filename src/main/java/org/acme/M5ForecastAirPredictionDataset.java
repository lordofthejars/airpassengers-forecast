package org.acme;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Date;


import com.google.gson.GsonBuilder;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.dataset.TimeSeriesDataset;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;

public class M5ForecastAirPredictionDataset extends TimeSeriesDataset {

    // String url = "https://resources.djl.ai/test-models/mxnet/timeseries/air_passengers.json";
    
    URL dataset;
    boolean train;

    // automatically calculated
    long datasetSize;
    AirPassengers passengers;

    public static class Builder extends TimeSeriesBuilder<Builder> {

        URL url;
        boolean train;

        @Override
        protected Builder self() {
            return this;
        }

        protected Builder train(boolean train) {
            this.train = train;
            return this;
        }

        protected Builder datasetJson(URL datasetJson) {
            this.url = datasetJson;
            return this;
        }

        protected M5ForecastAirPredictionDataset build() {
            return new M5ForecastAirPredictionDataset(this);
        }

    }

    public static Builder builder() {
        return new Builder();
    }

    protected M5ForecastAirPredictionDataset(Builder builder) {
        super(builder);
        this.dataset = builder.url;
        this.train = builder.train;
        this.datasetSize = 144 - 32;
    }

    @Override
    public void prepare(Progress progress) throws IOException, TranslateException {
        try (Reader reader = new InputStreamReader(this.dataset.openStream(), StandardCharsets.UTF_8)) {
            this.passengers =
                    new GsonBuilder()
                            .setDateFormat("yyyy-MM")
                            .create()
                            .fromJson(reader, AirPassengers.class);
        }
    }

    public static final class AirPassengers {
        Date start;
        float[] target;
    }

    @Override
    public TimeSeriesData getTimeSeriesData(NDManager manager, long index) {
            
            NDArray target = manager.create(passengers.target);
            long targetSize = target.size();
            NDList split = target.split(new long[]{targetSize - 32});
            TimeSeriesData data = new TimeSeriesData(10);

            if(this.train) {
                LocalDateTime start =
                    passengers.start.toInstant().atZone(ZoneId.systemDefault()).toLocalDateTime();
                data.setStartTime(start);

                NDArray testDataset = split.get(0);
                data.setField(FieldName.TARGET, testDataset);
                this.datasetSize = testDataset.size();
            } else {
                // Evaluation
                NDArray testDataset = split.get(1);
                data.setField(FieldName.TARGET, testDataset);
                this.datasetSize = testDataset.size();
            }

            return data;
    }

    @Override
    protected long availableSize() {
        return this.datasetSize;
    }
    
}
