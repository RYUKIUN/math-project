import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';

const ObesityPredictor = () => {
  // Model parameters
  const intercepts = [0.01139, 0.08290, 0.05336, -0.14764];
  const coefficients = [
    [-0.88801, 2.08869, -0.01944, 0.19546, -0.00101, 0.08236],
    [4.53598, -3.54045, -0.78216, 1.06957, 0.00238, 0.07184],
    [2.22910, -0.62322, -0.40478, 0.68897, -0.00122, 0.13058],
    [-5.87706, 2.07497, 1.20638, -1.95400, -0.00016, -0.28478]
  ];
  
  const classes = ['Normal', 'Obese', 'Overweight', 'Underweight'];
  
  // User inputs
  const [userInputs, setUserInputs] = useState({
    bmi: 25,
    age: 30,
    height: 170,
    weight: 70,
    daily_calories: 2000,
    sleep_hours: 7
  });
  
  const [adjustedValues, setAdjustedValues] = useState({
    daily_calories: 2000,
    sleep_hours: 7
  });
  
  const [prediction, setPrediction] = useState('');
  const [probabilities, setProbabilities] = useState([]);
  const [recommendation, setRecommendation] = useState({ calories: 0, sleep: 0 });

  // Calculate BMI from height and weight
  const calculateBMI = (height, weight) => {
    return weight / ((height / 100) ** 2);
  };

  // Softmax function
  const softmax = (logits) => {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(x => Math.exp(x - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map(x => x / sumExp);
  };

  // Predict class
  const predict = (features, calories = null, sleep = null) => {
    const actualFeatures = [
      features.bmi,
      features.age,
      features.height,
      features.weight,
      calories !== null ? calories : features.daily_calories,
      sleep !== null ? sleep : features.sleep_hours
    ];
    
    const logits = coefficients.map((coef, i) => {
      return intercepts[i] + coef.reduce((sum, c, j) => sum + c * actualFeatures[j], 0);
    });
    
    const probs = softmax(logits);
    const maxProbIndex = probs.indexOf(Math.max(...probs));
    
    return {
      prediction: classes[maxProbIndex],
      probabilities: probs,
      logits: logits
    };
  };

  // Find optimal calories and sleep for normal classification
  const findOptimalValues = () => {
    let bestCalories = userInputs.daily_calories;
    let bestSleep = userInputs.sleep_hours;
    let bestNormalProb = 0;
    
    // Grid search for optimal values
    for (let calories = 1200; calories <= 3500; calories += 50) {
      for (let sleep = 4; sleep <= 8; sleep += 0.5) {
        const result = predict(userInputs, calories, sleep);
        const normalProb = result.probabilities[0]; // Normal is index 0
        
        if (normalProb > bestNormalProb) {
          bestNormalProb = normalProb;
          bestCalories = calories;
          bestSleep = sleep;
        }
      }
    }
    
    return { calories: bestCalories, sleep: bestSleep, probability: bestNormalProb };
  };

  // Update prediction when inputs change
  useEffect(() => {
    const currentBMI = calculateBMI(userInputs.height, userInputs.weight);
    const updatedInputs = { ...userInputs, bmi: currentBMI };
    
    const result = predict(updatedInputs, adjustedValues.daily_calories, adjustedValues.sleep_hours);
    setPrediction(result.prediction);
    setProbabilities(result.probabilities);
    
    // Find optimal values for normal classification
    const optimal = findOptimalValues();
    setRecommendation(optimal);
  }, [userInputs, adjustedValues]);

  const handleInputChange = (field, value) => {
    setUserInputs(prev => ({
      ...prev,
      [field]: parseFloat(value)
    }));
  };

  const handleAdjustmentChange = (field, value) => {
    setAdjustedValues(prev => ({
      ...prev,
      [field]: parseFloat(value)
    }));
  };

  const currentBMI = calculateBMI(userInputs.height, userInputs.weight);

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-center">
            Obesity Classification Predictor & Optimizer
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Basic User Information */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Age</label>
              <input
                type="number"
                value={userInputs.age}
                onChange={(e) => handleInputChange('age', e.target.value)}
                className="w-full p-2 border rounded-md"
                min="1"
                max="120"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Height (cm)</label>
              <input
                type="number"
                value={userInputs.height}
                onChange={(e) => handleInputChange('height', e.target.value)}
                className="w-full p-2 border rounded-md"
                min="100"
                max="250"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Weight (kg)</label>
              <input
                type="number"
                value={userInputs.weight}
                onChange={(e) => handleInputChange('weight', e.target.value)}
                className="w-full p-2 border rounded-md"
                min="30"
                max="200"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">BMI (calculated)</label>
              <input
                type="number"
                value={currentBMI.toFixed(1)}
                readOnly
                className="w-full p-2 border rounded-md bg-gray-100"
              />
            </div>
          </div>

          {/* Adjustable Parameters */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Adjustable Parameters</h3>
            
            <div>
              <label className="block text-sm font-medium mb-2">
                Daily Calories: {adjustedValues.daily_calories}
              </label>
              <input
                type="range"
                min="1200"
                max="3500"
                step="50"
                value={adjustedValues.daily_calories}
                onChange={(e) => handleAdjustmentChange('daily_calories', e.target.value)}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">
                Sleep Hours: {adjustedValues.sleep_hours}
              </label>
              <input
                type="range"
                min="4"
                max="8"
                step="0.5"
                value={adjustedValues.sleep_hours}
                onChange={(e) => handleAdjustmentChange('sleep_hours', e.target.value)}
                className="w-full"
              />
            </div>
          </div>

          {/* Current Prediction */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-2">Current Prediction</h3>
            <div className="text-xl font-bold mb-2">
              Classification: <span className={`${
                prediction === 'Normal' ? 'text-green-600' : 
                prediction === 'Overweight' ? 'text-yellow-600' : 
                prediction === 'Obese' ? 'text-red-600' : 'text-blue-600'
              }`}>{prediction}</span>
            </div>
            
            <div className="space-y-1">
              {classes.map((className, index) => (
                <div key={className} className="flex justify-between items-center">
                  <span>{className}:</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${probabilities[index] * 100}%` }}
                      />
                    </div>
                    <span className="text-sm">{(probabilities[index] * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Recommendations */}
          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-2">Recommendations for Normal Classification</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Optimal Daily Calories:</span>
                <span className="font-semibold">{recommendation.calories}</span>
                <span className="text-sm text-gray-600">
                  ({recommendation.calories > adjustedValues.daily_calories ? '+' : ''}{recommendation.calories - adjustedValues.daily_calories})
                </span>
              </div>
              <div className="flex justify-between">
                <span>Optimal Sleep Hours:</span>
                <span className="font-semibold">{recommendation.sleep}</span>
                <span className="text-sm text-gray-600">
                  ({recommendation.sleep > adjustedValues.sleep_hours ? '+' : ''}{(recommendation.sleep - adjustedValues.sleep_hours).toFixed(1)})
                </span>
              </div>
              <div className="text-sm text-gray-600 mt-2">
                Expected Normal probability: {(recommendation.probability * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Quick Apply Button */}
          <button
            onClick={() => setAdjustedValues({
              daily_calories: recommendation.calories,
              sleep_hours: recommendation.sleep
            })}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors"
          >
            Apply Recommended Values
          </button>
        </CardContent>
      </Card>
    </div>
  );
};

export default ObesityPredictor;