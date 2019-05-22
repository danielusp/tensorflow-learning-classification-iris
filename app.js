const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

(async () => {
    // Creating a model to predict the output
    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [4],
        units: 5,
        activation: 'sigmoid'
    }));

    model.add(tf.layers.dense({
        inputShape: [5],
        units: 3,
        activation: 'sigmoid'
    }));

    model.add(tf.layers.dense({
        units: 3,
        activation: 'sigmoid'
    }));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.adam(.06)
    });

    // Building dataset
    const irisDataSet = JSON.parse(fs.readFileSync('data/iris.json'));
    const irisDataTest = JSON.parse(fs.readFileSync('data/iris-testing.json'));
    
    /**
     * Training data
     * 4 input units
     * 
     * [
     *  [5.1, 3.5, 1.4, 0.2],
     *  [4.9, 3, 1.4, 0.2],
     *  ...
     * ]
     */
    const trainingData = tf.tensor2d(irisDataSet.map(item => [
        item.sepal_length, 
        item.sepal_width, 
        item.petal_length, 
        item.petal_width
    ]));
    
    /**  
     *  Output data related with input
     *  3 output units
     *  
     *  setosa      => [1,0,0]
     *  virginica   => [0,1,0]
     *  versicolor  => [0,0,1]
     */
    const outputData = tf.tensor2d(irisDataSet.map(item => [
        item.species === "setosa" ? 1 : 0,
        item.species === "virginica" ? 1 : 0,
        item.species === "versicolor" ? 1 : 0
    ]));
    
    /**  
     *  Testing Data result must be near
     *  3 output units
     *  
     *  setosa      => [1,0,0]
     *  virginica   => [0,1,0]
     *  versicolor  => [0,0,1]
     */
    const testingData = tf.tensor2d(irisDataTest.map(item => [
        item.sepal_length, 
        item.sepal_width, 
        item.petal_length, 
        item.petal_width
    ]));

    // Train the model
    await model.fit(trainingData, outputData, {
        epochs: 500
    });

    /**
     * Predict
     * 
     * Must be near..
     * [
     *  [0.9931253, 0.0000479, 0.0046883],
     *  [0.0006025, 0.9449097, 0.0506826],
     *  [0.0075403, 0.0023434, 0.9908047]
     * ]
     */
    model.predict(testingData).print();

})();