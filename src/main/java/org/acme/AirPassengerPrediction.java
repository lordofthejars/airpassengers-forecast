package org.acme;

import ai.djl.timeseries.timefeature.TimeFeature;
import ai.djl.timeseries.translator.DeepARTranslator;
import ai.djl.training.dataset.Batch;
import ai.djl.util.Progress;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.Reader;
import java.net.URI;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

import com.google.gson.GsonBuilder;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.timeseries.Forecast;
import ai.djl.timeseries.SampleForecast;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.dataset.M5Forecast;
import ai.djl.timeseries.dataset.TimeFeaturizers;
import ai.djl.timeseries.dataset.TimeSeriesDataset;
import ai.djl.timeseries.distribution.DistributionLoss;
import ai.djl.timeseries.distribution.output.DistributionOutput;
import ai.djl.timeseries.distribution.output.NegativeBinomialOutput;
import ai.djl.timeseries.evaluator.Rmsse;
import ai.djl.timeseries.model.deepar.DeepARNetwork;
import ai.djl.timeseries.transform.TimeSeriesTransform;
import ai.djl.timeseries.transform.feature.Feature;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.DeferredTranslatorFactory;
import ai.djl.translate.TranslateException;
import io.quarkus.runtime.QuarkusApplication;
import io.quarkus.runtime.annotations.QuarkusMain;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@QuarkusMain
public class AirPassengerPrediction implements QuarkusApplication {

    @Override
    public int run(String... args) throws Exception {
        AirPassengerPrediction.runExample(args);
        Path loc = Paths.get( "output/model").toAbsolutePath();

        System.out.println(loc);
        predict(loc);
        return 0;
    }

    private void predict(Path outputDir)
        throws MalformedModelException, IOException, TranslateException, ModelNotFoundException {

        try (Model model = Model.newInstance("deepar");NDManager manager = NDManager.newBaseManager("MXNet")) {
            DeepARNetwork predictionNetwork = getDeepARModel("M", 12, new NegativeBinomialOutput(), false);
            model.setBlock(predictionNetwork);
            model.load(outputDir);

            M5ForecastAirPredictionDataset testSet =
                getDataset(
                    new ArrayList<>(),
                    predictionNetwork.getContextLength(),
                    Dataset.Usage.TEST);

            Map<String, Object> arguments = new ConcurrentHashMap<>();
            arguments.put("prediction_length", 12);
            arguments.put("freq", "M");
            arguments.put("use_" + FieldName.FEAT_DYNAMIC_REAL.name().toLowerCase(), false);
            arguments.put("use_" + FieldName.FEAT_STATIC_CAT.name().toLowerCase(), false);
            arguments.put("use_" + FieldName.FEAT_STATIC_REAL.name().toLowerCase(), false);

            DeepARTranslator translator = DeepARTranslator.builder(arguments).build();

            Progress progress = new ProgressBar();
            progress.reset("Inferring", testSet.size());

            String url = "https://resources.djl.ai/test-models/mxnet/timeseries/air_passengers.json";
            try (Predictor<TimeSeriesData, Forecast> predictor = model.newPredictor(translator)) {
                final TimeSeriesData input = getTimeSeriesData(manager, URI.create(url).toURL());
                NDArray target = input.get(FieldName.TARGET);
                target.setName("target");
                Forecast forecast = predictor.predict(input);

                NDArray prediction = forecast.mean();

                System.out.println(Arrays.toString(prediction.toFloatArray()));
            }
        }

    }

    private static TimeSeriesData getTimeSeriesData(NDManager manager, URL url) throws IOException {
        try (Reader reader = new InputStreamReader(url.openStream(), StandardCharsets.UTF_8)) {
            M5ForecastAirPredictionDataset.AirPassengers passengers =
                new GsonBuilder()
                    .setDateFormat("yyyy-MM")
                    .create()
                    .fromJson(reader, M5ForecastAirPredictionDataset.AirPassengers.class);

            LocalDateTime start =
                passengers.start.toInstant().atZone(ZoneId.systemDefault()).toLocalDateTime();
            NDArray target = manager.create(passengers.target);
            TimeSeriesData data = new TimeSeriesData(10);
            data.setStartTime(start);
            data.setField(FieldName.TARGET, target);
            return data;
        }
    }

    public static TrainingResult runExample(String[] arguments) throws IOException, TranslateException {
        
        try (Model model = Model.newInstance("deepar")) {
            // specify the model distribution output, for M5 case, NegativeBinomial best describe it
            DistributionOutput distributionOutput = new NegativeBinomialOutput();
            DefaultTrainingConfig config = setupTrainingConfig("output/model", 1, distributionOutput);

            NDManager manager = model.getNDManager();
            DeepARNetwork trainingNetwork = getDeepARModel("M", 12, distributionOutput, true);
            model.setBlock(trainingNetwork);

             List<TimeSeriesTransform> trainingTransformation =
                    trainingNetwork.createTrainingTransformation(manager);
            int contextLength = trainingNetwork.getContextLength();

            M5ForecastAirPredictionDataset trainSet =
                    getDataset(trainingTransformation, contextLength, Dataset.Usage.TRAIN);

            try ( Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                System.out.println("+++++" + trainSet.availableSize());
                int historyLength = trainingNetwork.getHistoryLength();
                Shape[] inputShapes = new Shape[4];
                // (N, num_cardinality)
                inputShapes[0] = new Shape(1, 1);
                // (N, num_real) if use_feat_stat_real else (N, 1)
                inputShapes[1] = new Shape(1, 1);

                inputShapes[2] =
                    new Shape(
                        1,
                        historyLength,
                        TimeFeature.timeFeaturesFromFreqStr("M").size() + 1);
                inputShapes[3] = new Shape(1, historyLength);

                trainer.initialize(inputShapes);

                EasyTrain.fit(trainer, 5, trainSet, null);
                return trainer.getTrainingResult();
            }

        }

    }

    private static DeepARNetwork getDeepARModel(String freq, int predictionLength,
            DistributionOutput distributionOutput, boolean training) {
        // here is feat_static_cat's cardinality which depend on your dataset, change to what need
        
        List<Integer> cardinality = new ArrayList<>();
        cardinality.add(144 - 32);

        DeepARNetwork.Builder builder =
                DeepARNetwork.builder()
                        .setFreq(freq)
                        .setPredictionLength(predictionLength)
                        .setCardinality(cardinality)
                        .optDistrOutput(distributionOutput)
                        .optUseFeatStaticCat(false);
        return training ? builder.buildTrainingNetwork() : builder.buildPredictionNetwork();
    }

    private static DefaultTrainingConfig setupTrainingConfig(
            String outputDir, int maxGpu, DistributionOutput distributionOutput) {
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float rmsse = result.getValidateEvaluation("RMSSE");
                    model.setProperty("RMSSE", String.format("%.5f", rmsse));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        return new DefaultTrainingConfig(new DistributionLoss("Loss", distributionOutput))
                .addEvaluator(new Rmsse(distributionOutput))
                .optDevices(Engine.getInstance().getDevices(maxGpu))
                .optInitializer(new XavierInitializer(), Parameter.Type.WEIGHT)
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }

    private static M5ForecastAirPredictionDataset getDataset(
            List<TimeSeriesTransform> transformation, int contextLength, Dataset.Usage usage)
            throws IOException {
            
            M5ForecastAirPredictionDataset.Builder builder = 
                M5ForecastAirPredictionDataset.builder()
                .datasetJson(URI.create("https://resources.djl.ai/test-models/mxnet/timeseries/air_passengers.json").toURL())
                .train(usage == Dataset.Usage.TRAIN)
                .setTransformation(transformation)
                .setContextLength(contextLength)
                .setSampling(32, usage == Dataset.Usage.TRAIN);
                
                return builder.build();
       
    }


}
