using System;
using Numpy;

namespace Neural03
{
    class Program
    {
        private const int NbInputsNeurons = 5;
        private const int NbHiddenNeurons = 14;
        private const int NbOutputNeurons = 1;
        private const int NbIterations = 10000;

        private static NDarray _trainSet;
        private static NDarray _trainExpected;
        private static NDarray _testSet;

        static NDarray _inputWeights;
        static NDarray _hiddenWeights;
        private static NDarray _inputLayer;
        private static NDarray _hiddenLayer;
        private static NDarray _outputLayer;

        static void Main(string[] args)
        {
            np.random.seed(1);
            InitializeDataset();
            InitializeWeights();

            for (var i = 0; i < NbIterations; i++)
            {
                FeedForward(_inputWeights);
                BackPropagation(i);
            }

            Predict();
        }

        private static void Predict()
        {
            var inputLayerTest = _testSet;
            var hiddenLayerTest = Sigmoid(np.dot(inputLayerTest, _inputWeights));
            var outputLayerTest = Sigmoid(np.dot(hiddenLayerTest, _hiddenWeights));
            
            LogTestPredicion(outputLayerTest);
        }

        private static void LogTestPredicion(NDarray outputLayerTest)
        {
            for (int j = 0; j < outputLayerTest.len; j++)
            {
                Console.WriteLine("prediction : " + Math.Round((double) outputLayerTest[j][0], 4));
            }
        }

        private static void InitializeWeights()
        {
            _inputWeights = 2 * np.random.rand(NbInputsNeurons, NbHiddenNeurons) - 1;
            _hiddenWeights = 2 * np.random.rand(NbHiddenNeurons, NbOutputNeurons) - 1;
        }
        
        private static void FeedForward(NDarray inputWeights)
        {
            _inputLayer = _trainSet;
            _hiddenLayer = Sigmoid(np.dot(_inputLayer, inputWeights));
            _outputLayer = Sigmoid(np.dot(_hiddenLayer, _hiddenWeights));
        }

        private static void BackPropagation(int i)
        {
            var errors = (_trainExpected - _outputLayer);
            var outputLayerDelta = errors * SigmoidPrime(_outputLayer);
            var hiddenLayerError = np.dot(outputLayerDelta, _hiddenWeights.T);
            var hiddenLayerDelta = hiddenLayerError * SigmoidPrime(_hiddenLayer);

            WeightsAdjustment(outputLayerDelta, hiddenLayerDelta);

            LogErrors(i, errors);
        }

        private static void WeightsAdjustment(NDarray outputLayerDelta, NDarray hiddenLayerDelta)
        {
            _hiddenWeights += np.dot(_hiddenLayer.T, outputLayerDelta);
            _inputWeights += np.dot(_inputLayer.T, hiddenLayerDelta);
        }

        private static void LogErrors(int i, NDarray errors)
        {
            if (i % 20 != 0) return;
            var cost = np.mean(np.abs(errors));
            Console.WriteLine("Cout:" + Math.Round(cost, 3));
        }

        private static void InitializeDataset()
        {
            _trainSet = np.array(new NDarray[]
            {
                new[] {1, 1, 1, 1, 1},
                new[] {0, 1, 1, 0, 1},
                new[] {0, 0, 1, 1, 0},
                new[] {0, 0, 1, 1, 1},
                new[] {1, 1, 1, 0, 1},
                new[] {1, 0, 1, 0, 0},
                new[] {1, 1, 0, 1, 1},
            });

            _trainExpected = np.array(new NDarray[]
            {
                new[] {1},
                new[] {0},
                new[] {0},
                new[] {1},
                new[] {0},
                new[] {0},
                new[] {1},
            });

            _testSet = np.array(new NDarray[]
            {
                new[] {0, 0, 0, 1, 1},
                new[] {0, 0, 1, 0, 0},
                new[] {1, 0, 0, 0, 0},
                new[] {0, 1, 1, 1, 1},
            });
        }

        private static NDarray Sigmoid(NDarray x)
        {
            return 1 / (1 + np.exp(-x));
        }

        private static NDarray SigmoidPrime(NDarray x)
        {
            return x * (1 - x);
        }
    }
}