package jp.weeelb;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
//import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.AlexNet;
import org.deeplearning4j.zoo.model.FaceNetNN4Small2;
import org.deeplearning4j.zoo.model.LeNet;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ImageClassificationLearning {
	private static final String DATADIR = "data";
	private static final String MODEL = "model.bin";
	private static final String LABEL = "label.txt";
	private static final String ES_MODEL_DIRECTORY = "earlystopping";

	private static int MAX_PATHS_PER_LABEL = 0;

	// input shape (channels, width, height)
	private static int[] LENET_INPUT_SHAPE = new int[] {3, 224, 224};
	private static int[] ALEXNET_INPUT_SHAPE = new int[] {3, 224, 224};
	private static int[] RESNET50_INPUT_SHAPE = new int[] {3, 224, 224};
	private static int[] FACENET_INPUT_SHAPE = new int[] {3, 96, 96};

	// shape index
	private static int CHANNELS = 0;
	private static int WIDTH = 1;
	private static int HEIGHT = 2;

	private static int BATCH_SIZE = 10;
	private static int EPOCHS = 10;
	private static double SPLIT_TRAIN_TEST = 0.8d;

    protected long seed = 42;
    protected Random rng = new Random(seed);

    protected List<String> labels;

    private ParentPathLabelGenerator labelMaker;
    private InputSplit trainData;
    private InputSplit testData;

    /**
     * LeNetによる学習を実行
     */
	public void learnLeNet() {
		setupData();
		learn(lenet(), LENET_INPUT_SHAPE);
	}

	/**
	 * AlexNetによる学習を実行
	 */
	public void learnAlexNet() {
		setupData();
		learn(alexnet(), ALEXNET_INPUT_SHAPE);
	}

	/**
	 * ResNet50による学習を実行
	 */
	public void learnResNet50() {
		setupData();
		learn(resnet50(), RESNET50_INPUT_SHAPE);
	}

	/**
	 * FaceNetによる学習を実行
	 */
	public void learnFaceNet() {
		setupData();
		learn(facenet(), FACENET_INPUT_SHAPE);
	}

    /**
     * LeNetによる学習を実行(アーリーストッピング)
     */
	public void learnLeNetES() {
		setupData();
		learnES(lenet(), LENET_INPUT_SHAPE);
	}

	/**
	 * AlexNetによる学習を実行(アーリーストッピング)
	 */
	public void learnAlexNetES() {
		setupData();
		learnES(alexnet(), ALEXNET_INPUT_SHAPE);
	}

	/**
	 * ResNet50による学習を実行(アーリーストッピング)
	 */
	public void learnResNet50ES() {
		setupData();
		learnES(resnet50(), RESNET50_INPUT_SHAPE);
	}

	/**
	 * FaceNetによる学習を実行(アーリーストッピング)
	 */
	public void learnFaceNetES() {
		setupData();
		learnES(facenet(), FACENET_INPUT_SHAPE);
	}

	/**
	 * データセットアップ
	 */
	private void setupData() {
		message("Data directory: " + DATADIR);
		File mainPath = new File(DATADIR);
		FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
		labelMaker = new ParentPathLabelGenerator();

		// ラベル取得
		getLabels(fileSplit.getRootDir());

		int numExamples = java.lang.Math.toIntExact(fileSplit.length());
		BalancedPathFilter pathFilter = new BalancedPathFilter(
				rng, labelMaker, numExamples, labels.size(), MAX_PATHS_PER_LABEL);

		// トレーニング用と評価用に分割
		InputSplit[] inputSplit = fileSplit.sample(pathFilter, SPLIT_TRAIN_TEST, 1 - SPLIT_TRAIN_TEST);
		trainData = inputSplit[0];
		testData = inputSplit[1];
	}

	/**
	 * 学習(MultiLayerNetwork)
	 *
	 * @param conf モデル定義
	 * @param inputShape 入力サイズ
	 */
	private void learn(MultiLayerConfiguration conf, int[] inputShape) {
		message("Input shape: " + inputShape[0] + ", " + inputShape[1] + ", " + inputShape[2]);

		// ニューラルネットワーク構築
		MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        try(ImageRecordReader recordReader = new ImageRecordReader(
        		inputShape[HEIGHT], inputShape[WIDTH], inputShape[CHANNELS], labelMaker)) {
        	// 学習状況をモニタリング(http://localhost:9000)
	        UIServer uiServer = UIServer.getInstance();
	        StatsStorage statsStorage = new InMemoryStatsStorage();
	        uiServer.attach(statsStorage);
	        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));

			// 画像データを0〜1の実数に正規化
	        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

	        // トレーニング
	        recordReader.initialize(trainData, null);
	        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, BATCH_SIZE, 1, labels.size());
	        scaler.fit(dataIter);
	        dataIter.setPreProcessor(scaler);
	        network.fit(dataIter, EPOCHS);

	        // 評価
	        recordReader.initialize(testData);
	        dataIter = new RecordReaderDataSetIterator(recordReader, BATCH_SIZE, 1, labels.size());
	        scaler.fit(dataIter);
	        dataIter.setPreProcessor(scaler);
	        Evaluation eval = network.evaluate(dataIter);
	        message(eval.stats(true));
        } catch (Exception e) {
        	e.printStackTrace();
        }

        try {
        	// 学習済みモデル保存
			ModelSerializer.writeModel(network, MODEL, true);
		} catch (IOException e) {
        	e.printStackTrace();
		}

        // ラベル保存
        saveLabel();

        message("Finished");
	}

	/**
	 * 学習(ComputationGraph)
	 *
	 * @param conf モデル定義
	 * @param inputShape 入力サイズ
	 */
	private void learn(ComputationGraph graph, int[] inputShape) {
		message("Input shape: " + inputShape[0] + ", " + inputShape[1] + ", " + inputShape[2]);

		// ニューラルネットワーク構築
        graph.init();

        try(ImageRecordReader recordReader = new ImageRecordReader(
        		inputShape[HEIGHT], inputShape[WIDTH], inputShape[CHANNELS], labelMaker)) {
        	// 学習状況をモニタリング(http://localhost:9000)
	        UIServer uiServer = UIServer.getInstance();
	        StatsStorage statsStorage = new InMemoryStatsStorage();
	        uiServer.attach(statsStorage);
	        graph.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));

			// 画像データを0〜1の実数に正規化
	        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

	        // トレーニング
	        recordReader.initialize(trainData, null);
	        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, BATCH_SIZE, 1, labels.size());
	        scaler.fit(dataIter);
	        dataIter.setPreProcessor(scaler);
	        graph.fit(dataIter, EPOCHS);

	        // 評価
	        recordReader.initialize(testData);
	        dataIter = new RecordReaderDataSetIterator(recordReader, BATCH_SIZE, 1, labels.size());
	        scaler.fit(dataIter);
	        dataIter.setPreProcessor(scaler);
	        Evaluation eval = graph.evaluate(dataIter);
	        message(eval.stats(true));
        } catch (Exception e) {
        	e.printStackTrace();
        }

        try {
        	// 学習済みモデル保存
			ModelSerializer.writeModel(graph, MODEL, true);
		} catch (IOException e) {
			e.printStackTrace();
		}

        // ラベル保存
        saveLabel();

        message("Finished");
	}

	/**
	 * 学習(MultiLayerNetwork - アーリーストッピング)
	 *
	 * @param conf モデル定義
	 * @param inputShape 入力サイズ
	 */
	private void learnES(MultiLayerConfiguration networkConfig, int[] inputShape) {
		message("Input shape: " + inputShape[0] + ", " + inputShape[1] + ", " + inputShape[2]);
        try(ImageRecordReader trainReader = new ImageRecordReader(
        		inputShape[HEIGHT], inputShape[WIDTH], inputShape[CHANNELS], labelMaker);
        		ImageRecordReader testReader = new ImageRecordReader(
        				inputShape[HEIGHT], inputShape[WIDTH], inputShape[CHANNELS], labelMaker)) {
    		// 画像データを0〜1の実数に正規化
    		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

    		// トレーニング用データ
    		trainReader.initialize(trainData, null);
        	DataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader, BATCH_SIZE, 1, labels.size());
        	trainIter.setPreProcessor(scaler);

        	// 評価用データ
        	testReader.initialize(testData, null);
        	DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, BATCH_SIZE, 1, labels.size());
        	testIter.setPreProcessor(scaler);

        	// アーリーストッピングの設定(停止条件はエポック数)
	        message("Build model....");
        	EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
        			.epochTerminationConditions(new MaxEpochsTerminationCondition(EPOCHS))
        			.scoreCalculator(new DataSetLossCalculator(testIter, true))
        	        .evaluateEveryNEpochs(1)
        			.modelSaver(new LocalFileModelSaver(ES_MODEL_DIRECTORY))
        			.build();
        	EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, networkConfig, trainIter);

        	// ラベル保存
            saveLabel();

	        // トレーニング
        	EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

        	// 学習済みモデル保存(最もスコアの良いモデル)
        	Model bestModel = result.getBestModel();
            ModelSerializer.writeModel(bestModel, MODEL, true);

            message("Finished");
        } catch (Exception e) {
        	e.printStackTrace();
        }
	}

	/**
	 * 学習(ComputationGraph - アーリーストッピング)
	 *
	 * @param conf モデル定義
	 * @param inputShape 入力サイズ
	 */
	private void learnES(ComputationGraph graphConfig, int[] inputShape) {
		message("Input shape: " + inputShape[0] + ", " + inputShape[1] + ", " + inputShape[2]);

        message("Data Setup -> define how to load data into net");
        try(ImageRecordReader trainReader = new ImageRecordReader(
        		inputShape[HEIGHT], inputShape[WIDTH], inputShape[CHANNELS], labelMaker);
        		ImageRecordReader testReader = new ImageRecordReader(
        				inputShape[HEIGHT], inputShape[WIDTH], inputShape[CHANNELS], labelMaker)) {
    		// 画像データを0〜1の実数に正規化
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

    		// トレーニング用データ
            trainReader.initialize(trainData, null);
        	DataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader, BATCH_SIZE, 1, labels.size());
        	trainIter.setPreProcessor(scaler);

        	// 評価用データ
        	testReader.initialize(testData, null);
        	DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, BATCH_SIZE, 1, labels.size());
        	testIter.setPreProcessor(scaler);

        	// アーリーストッピングの設定(停止条件はエポック数)
        	EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
        			.epochTerminationConditions(new MaxEpochsTerminationCondition(EPOCHS))
        			.scoreCalculator(new DataSetLossCalculator(testIter, true))
        	        .evaluateEveryNEpochs(1)
        			.modelSaver(new LocalFileGraphSaver(ES_MODEL_DIRECTORY))
        			.build();
        	EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf, graphConfig, trainIter);

        	// ラベル保存
            saveLabel();

	        // トレーニング
        	EarlyStoppingResult<ComputationGraph> result = trainer.fit();

        	// 学習済みモデル保存(最もスコアの良いモデル)
        	Model bestModel = result.getBestModel();
            ModelSerializer.writeModel(bestModel, MODEL, true);

            message("Finished");
        } catch (Exception e) {
        	e.printStackTrace();
        }
	}

	/**
	 * モデル定義(LeNet)
	 */
	private MultiLayerConfiguration lenet0() {
		return new NeuralNetConfiguration.Builder().seed(seed)
                .activation(Activation.IDENTITY)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new AdaDelta())
                .cacheMode(CacheMode.NONE)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                .convolutionMode(ConvolutionMode.Same)
                .list()
                // block 1
                .layer(0, new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {1, 1}).name("cnn1")
                                .nIn(LENET_INPUT_SHAPE[0]).nOut(20).activation(Activation.RELU).build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2},
                                new int[] {2, 2}).name("maxpool1").build())
                // block 2
                .layer(2, new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {1, 1}).name("cnn2").nOut(50)
                                .activation(Activation.RELU).build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2},
                                new int[] {2, 2}).name("maxpool2").build())
                // fully connected
                .layer(4, new DenseLayer.Builder().name("ffn1").activation(Activation.RELU).nOut(500).build())
                // output
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).name("output")
                                .nOut(labels.size()).activation(Activation.SOFTMAX) // radial basis function required
                                .build())
                .setInputType(InputType.convolutionalFlat(LENET_INPUT_SHAPE[2], LENET_INPUT_SHAPE[1], LENET_INPUT_SHAPE[0]))
                .build();
	}

	/**
	 * モデル定義(LeNet - Zoo Model)
	 */
	private MultiLayerConfiguration lenet() {
		return LeNet.builder()
				.numClasses(labels.size())
				.seed(seed)
				.build()
				.conf();
	}

	/**
	 * モデル定義(AlexNet)
	 */
	@SuppressWarnings("deprecation")
	private MultiLayerConfiguration alexnet0() {
        double nonZeroBias = 1;
        return new NeuralNetConfiguration.Builder().seed(seed)
                        .weightInit(new NormalDistribution(0.0, 0.01))
                        .activation(Activation.RELU)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Nesterovs(1e-2, 0.9))
                        .biasUpdater(new Nesterovs(2e-2, 0.9))
                        .convolutionMode(ConvolutionMode.Same)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                        .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                        .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                        .cacheMode(CacheMode.NONE)
                        .l2(5 * 1e-4)
                        .miniBatch(false)
                        .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{11,11}, new int[]{4, 4})
                        .name("cnn1")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .nIn(ALEXNET_INPUT_SHAPE[0])
                        .nOut(96)
                        .build())
                .layer(1, new LocalResponseNormalization.Builder().build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .padding(1,1)
                        .name("maxpool1")
                        .build())
                .layer(3, new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, new int[]{2,2})
                        .name("cnn2")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{2, 2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .name("maxpool2")
                        .build())
                .layer(5, new LocalResponseNormalization.Builder().build())
                .layer(6, new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .convolutionMode(ConvolutionMode.Same)
                        .name("cnn3")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(384)
                        .build())
                .layer(7, new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1})
                        .name("cnn4")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(384)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(8, new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1})
                        .name("cnn5")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(9, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2})
                        .name("maxpool3")
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build())
                .layer(10, new DenseLayer.Builder()
                        .name("ffn1")
                        .nIn(256*6*6)
                        .nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .build())
                .layer(11, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(4096)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .dropOut(0.5)
                        .build())
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(labels.size())
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.005))
                        .biasInit(0.1)
                        .build())
                .setInputType(InputType.convolutional(ALEXNET_INPUT_SHAPE[2], ALEXNET_INPUT_SHAPE[1], ALEXNET_INPUT_SHAPE[0]))
                .build();
	}

	/**
	 * モデル定義(AlexNet - Zoo Model)
	 */
	private MultiLayerConfiguration alexnet() {
		return AlexNet.builder()
				.numClasses(labels.size())
				.seed(seed)
				.build()
				.conf();
	}

	/**
	 * モデル定義(ResNet50 - Zoo Model)
	 */
	private ComputationGraph resnet50() {
		return ResNet50.builder()
				.numClasses(labels.size())
				.seed(seed)
				.build()
				.init();
	}

	/**
	 * モデル定義(FaceNetNN4Small2 - Zoo Model)
	 */
	private ComputationGraph facenet() {
		return FaceNetNN4Small2.builder()
				.numClasses(labels.size())
				.seed(seed)
				.build()
				.init();
	}

	/**
	 * ラベル取得
	 */
	private void getLabels(File root) {
		this.labels = Arrays.stream(root.listFiles(File::isDirectory))
				.map(file->file.getName())
				.collect(Collectors.toList());
	}

	/**
	 * ラベル本
	 */
	private void saveLabel() {
		try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(LABEL))) {
			for (String label : labels) {
				writer.write(label);
				writer.newLine();
			}
			writer.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * メッセージ出力
	 */
	private static void message(String message) {
		System.out.println(message);
	}

	/**
	 * エントリーポイント
	 */
	public static void main(String[] args) {
		ImageClassificationLearning icl = new ImageClassificationLearning();

//		icl.learnLeNet();		// 224x224
		icl.learnAlexNet();		// 224x224
//		icl.learnResNet50();	// 224x224
//		icl.learnFaceNet();		// 96x96
//		icl.learnLeNetES();		// 224x224
//		icl.learnAlexNetES();	// 224x224
//		icl.learnResNet50ES();	// 224x224
//		icl.learnFaceNetES();	// 96x96
	}
}
