import React, { useState, useEffect } from 'react';
import { Play, CheckCircle, XCircle } from 'lucide-react';
import * as ort from 'onnxruntime-web';

const CONFIG = {
  modelPath: 'facial_attributes_model.onnx',
  targetImages: 24,
  imageFolder: 'images_224',
  attributeNames: [
    'a man',   //'Gender'
    'a woman',
    'a boy',
    'a girl',

    'a person with blond hair',  //'Hair Color'
    'a person with brown hair',
    'a person with black hair',
    'a person with red hair',
    'a person with gray hair',
    'a person with white hair',
    'a bald person',

    'a bald person',           //'Hair Length'
    'a person with short hair',
    'a person with hair around his neck',

    'a bald person',              //'Hair Type'
    'a person with straight hair',
    'a person with curly hair',
    'a person with wavy hair',
    'a person with afro hair',

    'clean shaven',     //'Facial Hair'
    'stubble',
    'beard',

    'a person with brown eyes', //'Eye Color'
    'a person with blue eyes',
    'a person with green eyes',
    'a person with hazel eyes',
    'a person with black eyes',
    'a person with amber eyes',

    'a person with clean hair', //'Hat'
    'a person with few hair',
    'a woman with her head covered',
    'a woman with visible forehead',
    'a woman with a headband on her forehead',
    'a person with a hat',

    'a person with eyes',  //'Eyeglasses'
    'a person with visible eyes',
    'a person with eye wrinkles',
    'a person with eye bags',
    'a person with eyeglasses'
  ],
  // Gruppi mutualmente esclusivi
  attributeGroups: [
    { name: 'Gender', indices: [0, 1, 2, 3] },
    { name: 'Hair Color', indices: [4, 5, 6, 7, 8, 9, 10] },
    { name: 'Hair Length', indices: [11, 12, 13] },
    { name: 'Hair Type', indices: [14, 15, 16, 17, 18] },
    { name: 'Facial Hair', indices: [19, 20, 21] },
    { name: 'Eye Color', indices: [22, 23, 24, 25, 26, 27] },
    { name: 'Hat', indices: [28, 29, 30, 31, 32, 33] },
    { name: 'Eyeglasses', indices: [34, 35, 36, 37, 38] }
  ],
  // Mapping per stampa personalizzata quando showDominantOnly è attivo
  displayMapping: {
    // Gender (0-3)
    0: 'Uomo',
    1: 'Donna',
    2: 'Uomo',
    3: 'Donna',
    
    // Hair Color (4-10)
    4: 'Capelli biondi',
    5: 'Capelli Marroni',
    6: 'Capelli neri',
    7: 'Capelli rossi',
    8: 'Capelli grigi',
    9: 'Capelli bianchi',
    10: 'No Capelli',
    
    // Hair Length (11-13)
    11: 'No capelli',
    12: 'Capelli corti',
    13: 'Capelli lunghi',
    
    // Hair Type (14-18)
    14: 'No capelli',
    15: 'Capelli lisci',
    16: 'Capelli ricci',
    17: 'Capelli mossi',//questi due possono rientrare in ricci
    18: 'Capelli afro',
    
    // Facial Hair (19-21)
    19: 'Senza Barba', 
    20: 'Con Barba',
    21: 'Con Barba',
    
    // Eye Color (22-27)
    22: 'Occhi marroni', //'Brown Eyes',
    23: 'Occhi azzurri',
    24: 'Occhi verdi',
    25: 'Occhi verdi',
    26: 'Occhi marroni',
    27: 'Occhi verdi',
    
    // Hat (28-33)
    28: 'Senza Cappello',
    29: 'Senza Cappello',
    30: 'Con Cappello',
    31: 'Senza Cappello',
    32: 'Con Cappello',
    33: 'Con Cappello',
    
    // Eyeglasses (34-38)
    34: 'Senza Occhiali',
    35: 'Senza Occhiali',
    36: 'Senza Occhiali',
    37: 'Senza Occhiali',
    38: 'Con Occhiali'
  }
};

function FacialAttributesClassifier() {
  const [session, setSession] = useState(null);
  const [results, setResults] = useState([]);
  const [currentImage, setCurrentImage] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [showOnlyHighConfidence, setShowOnlyHighConfidence] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [showDominantOnly, setShowDominantOnly] = useState(true);
  const [attributeStats, setAttributeStats] = useState({});

  // Carica il modello ONNX
  const loadModel = async () => {
    try {
      setError(null);
      
      const modelResponse = await fetch('/facial_attributes_model.onnx');
      const modelBuffer = await modelResponse.arrayBuffer();
      
      const dataResponse = await fetch('/facial_attributes_model.onnx.data');
      const dataBuffer = await dataResponse.arrayBuffer();
      
      const sess = await ort.InferenceSession.create(modelBuffer, {
        externalData: [
          {
            data: dataBuffer,
            path: 'facial_attributes_model.onnx.data'
          }
        ]
      });
      
      setSession(sess);
      setModelLoaded(true);
      console.log('Modello caricato con successo');
    } catch (err) {
      setError(`Errore nel caricamento del modello: ${err.message}`);
      console.error(err);
    }
  };

  const preprocessImage = async (imageUrl) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas');
          canvas.width = 224;
          canvas.height = 224;
          const ctx = canvas.getContext('2d');
          
          ctx.drawImage(img, 0, 0, 224, 224);
          const imageData = ctx.getImageData(0, 0, 224, 224);
          const { data } = imageData;
          
          const red = [], green = [], blue = [];
          
          for (let i = 0; i < data.length; i += 4) {
            red.push(data[i] / 255.0);
            green.push(data[i + 1] / 255.0);
            blue.push(data[i + 2] / 255.0);
          }
          
          const input = new Float32Array([...red, ...green, ...blue]);
          const tensor = new ort.Tensor('float32', input, [1, 3, 224, 224]);
          
          resolve(tensor);
        } catch (err) {
          reject(err);
        }
      };
      
      img.onerror = () => reject(new Error(`Impossibile caricare l'immagine: ${imageUrl}`));
      img.src = imageUrl;
    });
  };

  const classifyImage = async (imageNumber) => {
    try {
      const paddedNumber = String(imageNumber).padStart(6, '0');
      const imageUrl = `${CONFIG.imageFolder}/${paddedNumber}.png`;
      
      const tensor = await preprocessImage(imageUrl);
      
      const feeds = {};
      feeds[session.inputNames[0]] = tensor;
      
      const output = await session.run(feeds);
      const predictions = output[session.outputNames[0]].data;
      
      const allAttributes = [];
      for (let i = 0; i < predictions.length && i < CONFIG.attributeNames.length; i++) {
        const probability = 1 / (1 + Math.exp(-predictions[i]));
        allAttributes.push({
          name: CONFIG.attributeNames[i],
          probability: probability,
          rawValue: predictions[i],
          index: i
        });
      }
      
      return {
        imageNumber,
        imageUrl,
        attributes: allAttributes,
        success: true
      };
    } catch (err) {
      return {
        imageNumber,
        imageUrl: `${CONFIG.imageFolder}/${String(imageNumber).padStart(6, '0')}.png`,
        error: err.message,
        success: false
      };
    }
  };

  const processAllImages = async () => {
    if (!session) {
      setError('Carica prima il modello');
      return;
    }
    
    setIsProcessing(true);
    setResults([]);
    setError(null);
    setElapsedTime(0);
    setAttributeStats({});
    
    const start = Date.now();
    const allResults = [];
    
    for (let i = 1; i <= CONFIG.targetImages; i++) {
      setCurrentImage(i);
      const result = await classifyImage(i);
      allResults.push(result);
      setResults(prev => [...prev, result]);
    }
    
    // Calcola le statistiche degli attributi dominanti
    const stats = {};
    allResults.forEach(result => {
      if (result.success) {
        const dominantAttrs = getDominantAttributes(result.attributes);
        dominantAttrs.forEach(attr => {
          const key = attr.displayName || attr.name;
          // Conta solo una volta per immagine anche se appare più volte
          if (!stats[key]) {
            stats[key] = new Set();
          }
          stats[key].add(result.imageNumber);
        });
      }
    });
    
    // Converti i Set in conteggi
    const finalStats = {};
    Object.keys(stats).forEach(key => {
      finalStats[key] = stats[key].size;
    });
    
    setAttributeStats(finalStats);
    
    const totalTime = ((Date.now() - start) / 1000).toFixed(2);
    setElapsedTime(totalTime);
    setIsProcessing(false);
    setCurrentImage(0);
  };

  // Filtra attributi mostrando solo il dominante per gruppo
  const getDominantAttributes = (attributes) => {
    const dominantAttrs = [];
    const usedIndices = new Set();
    
    // Per ogni gruppo, trova l'attributo con probabilità maggiore
    CONFIG.attributeGroups.forEach(group => {
      let maxProb = -1;
      let maxAttr = null;
      
      group.indices.forEach(idx => {
        const attr = attributes.find(a => a.index === idx);
        if (attr && attr.probability > maxProb) {
          maxProb = attr.probability;
          maxAttr = attr;
        }
      });
      
      if (maxAttr) {
        // Usa il mapping per la visualizzazione se disponibile
        const displayName = CONFIG.displayMapping[maxAttr.index] || maxAttr.name;
        dominantAttrs.push({
          ...maxAttr,
          displayName: displayName
        });
        group.indices.forEach(idx => usedIndices.add(idx));
      }
    });
    
    // Aggiungi gli attributi che non fanno parte di nessun gruppo
    attributes.forEach(attr => {
      if (!usedIndices.has(attr.index)) {
        const displayName = CONFIG.displayMapping[attr.index] || attr.name;
        dominantAttrs.push({
          ...attr,
          displayName: displayName
        });
      }
    });
    
    return dominantAttrs;
  };

  const getFilteredAttributes = (attributes) => {
    let filtered = attributes;
    
    if (showDominantOnly) {
      filtered = getDominantAttributes(filtered);
    }
    
    if (showOnlyHighConfidence) {
      filtered = filtered.filter(attr => attr.probability > 0.5);
    }
    
    return filtered;
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">
            Facial Attributes Classifier
          </h1>
          
          <div className="flex flex-wrap items-center gap-3 mb-4">
            <button
              onClick={loadModel}
              disabled={modelLoaded || isProcessing}
              className="px-5 py-2.5 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              {modelLoaded ? '✓ Model Loaded' : 'Load Model'}
            </button>
            
            <button
              onClick={processAllImages}
              disabled={!modelLoaded || isProcessing}
              className="px-5 py-2.5 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              <Play size={18} />
              {isProcessing ? `Processing ${currentImage}/${CONFIG.targetImages}...` : 'Start Classification'}
            </button>

            {results.length > 0 && (
              <>
                <label className="flex items-center gap-2 cursor-pointer ml-4">
                  <input
                    type="checkbox"
                    checked={showDominantOnly}
                    onChange={(e) => setShowDominantOnly(e.target.checked)}
                    className="w-4 h-4 text-indigo-600 rounded focus:ring-2 focus:ring-indigo-500"
                  />
                  <span className="text-sm text-gray-700">Show dominant only</span>
                </label>
                
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showOnlyHighConfidence}
                    onChange={(e) => setShowOnlyHighConfidence(e.target.checked)}
                    className="w-4 h-4 text-indigo-600 rounded focus:ring-2 focus:ring-indigo-500"
                  />
                  <span className="text-sm text-gray-700">Only {'>'}50%</span>
                </label>
              </>
            )}
          </div>
          
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
              <strong>Error:</strong> {error}
            </div>
          )}
          
          {isProcessing && (
            <div className="bg-blue-50 border border-blue-200 px-4 py-3 rounded-lg">
              <div className="flex items-center gap-3 mb-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <span className="text-blue-700 text-sm font-medium">
                  Processing: {currentImage}/{CONFIG.targetImages}
                </span>
              </div>
              <div className="bg-blue-200 rounded-full h-1.5 overflow-hidden">
                <div 
                  className="bg-blue-600 h-full transition-all duration-300"
                  style={{ width: `${(currentImage / CONFIG.targetImages) * 100}%` }}
                ></div>
              </div>
            </div>
          )}

          {elapsedTime > 0 && !isProcessing && (
            <div className="bg-green-50 border border-green-200 px-4 py-3 rounded-lg">
              <div className="flex items-center gap-2">
                <CheckCircle size={18} className="text-green-600" />
                <span className="text-green-700 text-sm font-medium">
                  Completed in {elapsedTime} seconds
                </span>
              </div>
            </div>
          )}
        </div>

        {Object.keys(attributeStats).length > 0 && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Statistiche Attributi Dominanti</h2>
            <div className="bg-gray-50 rounded-lg p-4 font-mono text-sm">
              <pre className="whitespace-pre-wrap">
                {Object.entries(attributeStats)
                  .sort((a, b) => b[1] - a[1])
                  .map(([attribute, count], index) => {
                    const percentage = ((count / CONFIG.targetImages) * 100).toFixed(0);
                    const position = String(index + 1).padStart(2, ' ');
                    const attributePadded = attribute.padEnd(25, ' ');
                    return `${position}. ${attributePadded} ${count} (${percentage}%)`;
                  })
                  .join('\n')}
              </pre>
            </div>
          </div>
        )}
        
        {results.length > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {results.map((result) => {
              const filteredAttributes = getFilteredAttributes(result.attributes);
              return (
                <div key={result.imageNumber} className="bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow overflow-hidden">
                  <div className="relative">
                    <img 
                      src={result.imageUrl} 
                      alt={`${result.imageNumber}`}
                      className="w-full aspect-square object-cover"
                      onError={(e) => {
                        e.target.style.display = 'none';
                        e.target.nextSibling.style.display = 'flex';
                      }}
                    />
                    <div className="hidden w-full aspect-square bg-gray-100 items-center justify-center text-gray-400 flex-col gap-2">
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                        <polyline points="21 15 16 10 5 21"></polyline>
                      </svg>
                      <span className="text-xs">{String(result.imageNumber).padStart(6, '0')}.png</span>
                    </div>
                    <div className="absolute top-1.5 right-1.5 bg-white/90 backdrop-blur-sm px-2 py-0.5 rounded text-xs font-semibold text-gray-700">
                      {String(result.imageNumber).padStart(6, '0')}
                    </div>
                  </div>
                  
                  <div className="p-3">
                    {result.success ? (
                      <div className="space-y-1.5">
                        {filteredAttributes.map((attr, idx) => (
                          <div key={idx} className="flex items-center justify-between gap-2 text-xs">
                            <span className="text-gray-700 truncate flex-1">
                              {showDominantOnly && attr.displayName ? attr.displayName : attr.name}
                            </span>
                            <span className={`font-semibold flex-shrink-0 ml-2 ${
                              attr.probability > 0.7 ? 'text-green-600' : 
                              attr.probability > 0.5 ? 'text-blue-600' : 
                              'text-gray-500'
                            }`}>
                              {(attr.probability * 100).toFixed(0)}%
                            </span>
                          </div>
                        ))}
                        {filteredAttributes.length === 0 && (
                          <p className="text-xs text-gray-400 text-center py-2">No attributes match filters</p>
                        )}
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-red-600 text-xs">
                        <XCircle size={14} />
                        <span>Error</span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
        
        {!isProcessing && results.length === 0 && modelLoaded && (
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-12 text-center">
            <svg className="mx-auto text-gray-300 mb-4" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <circle cx="8.5" cy="8.5" r="1.5"></circle>
              <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
            <p className="text-gray-500">
              Ready to analyze images. Click "Start Classification" to begin.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default FacialAttributesClassifier;