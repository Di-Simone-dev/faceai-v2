import React, { useState, useEffect, type SetStateAction } from 'react';
import { Play, CheckCircle, XCircle } from 'lucide-react';
import * as ort from 'onnxruntime-web';

/**
 * CONFIGURAZIONE GLOBALE
 * Contiene tutti i parametri necessari per il funzionamento del classificatore:
 * - Percorsi dei file
 * - Nomi degli attributi facciali
 * - Gruppi di attributi mutualmente esclusivi
 * - Mappature per la visualizzazione in italiano
 */
const CONFIG = {
  // Percorso del file del modello ONNX dentro a public
  modelPath: 'model.onnx',
  
  // Numero totale di immagini da processare (24 per il gioco)
  targetImages: 24,
  
  // Cartella contenente le immagini da analizzare (risoluzione 256x256)
  imageFolder: 'images_256',
  
  /**
   * NOMI DEGLI ATTRIBUTI
   * Array di 18 descrizioni testuali per ciascun attributo che il modello può rilevare.
   * L'ordine corrisponde agli output del modello ONNX.
   */
  attributeNames: [
    // Smile (indice 0)
    'smile',
    
    // Genere (indici 1-2)
    'gender_male',
    'gender_female',
    
    // Colore capelli (indici 3-6)
    'hair_brown',
    'hair_black',
    'hair_blond',
    'hair_gray',
    
    // Lunghezza capelli (indici 7-8)
    'hair_long',
    'hair_short',
    
    // Etnia (indici 9-12)
    'ethnicity_asian',
    'ethnicity_black',
    'ethnicity_latino',
    'ethnicity_white',
    
    // Colore occhi (indici 13-15)
    'eye_blue',
    'eye_brown',
    'eye_green',
    
    // Peli facciali (indice 16)
    'has_facial_hair',
    
    // Occhiali (indice 17)
    'a person with eyeglasses'
  ],
  
  /**
   * GRUPPI DI ATTRIBUTI MUTUALMENTE ESCLUSIVI
   * Definisce quali attributi appartengono alla stessa categoria.
   * In ogni gruppo, solo l'attributo con la probabilità più alta viene considerato dominante.
   */
  attributeGroups: [
    { name: 'Gender', indices: [1, 2] },
    { name: 'Hair Color', indices: [3, 4, 5, 6] },
    { name: 'Hair Length', indices: [7, 8] },
    { name: 'Ethnicity', indices: [9, 10, 11, 12] },
    { name: 'Eye Color', indices: [13, 14, 15] }
  ],
  
  /**
   * MAPPATURA PER LA VISUALIZZAZIONE
   * Converte gli indici degli attributi in etichette leggibili in italiano.
   * Utilizzato quando showDominantOnly è attivo.
   */
  displayMapping: {
    // Smile (0)
    0: 'Sorriso',
    
    // Genere (1-2)
    1: 'Uomo',
    2: 'Donna',
    
    // Colore capelli (3-6)
    3: 'Capelli Marroni',
    4: 'Capelli Neri',
    5: 'Capelli Biondi',
    6: 'Capelli Grigi',
    
    // Lunghezza capelli (7-8)
    7: 'Capelli Lunghi',
    8: 'Capelli Corti',
    
    // Etnia (9-12)
    9: 'Asiatico',
    10: 'Nero',
    11: 'Latino',
    12: 'Bianco',
    
    // Colore occhi (13-15)
    13: 'Occhi Azzurri',
    14: 'Occhi Marroni',
    15: 'Occhi Verdi',
    
    // Peli facciali (16)
    16: 'Con Barba',
    
    // Occhiali (17)
    17: 'Con Occhiali'
  }
};

/**
 * COMPONENTE PRINCIPALE: FacialAttributesClassifier
 * 
 * Questo componente React implementa un classificatore di attributi facciali usando ONNX Runtime.
 * Permette di:
 * 1. Caricare un modello ONNX pre-addestrato
 * 2. Processare batch di immagini (24 immagini di default)
 * 3. Classificare gli attributi facciali per ogni immagine
 * 4. Visualizzare i risultati con statistiche aggregate
 */
function FacialAttributesClassifier() {
  // ====== STATI DEL COMPONENTE ======
  
  /**
   * session: Istanza della sessione ONNX Runtime
   * Null finché il modello non viene caricato con successo
   */
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  
  /**
   * results: Array contenente i risultati della classificazione per ogni immagine
   * Ogni elemento contiene:
   * - imageNumber: numero progressivo dell'immagine
   * - imageUrl: percorso dell'immagine
   * - attributes: array di attributi rilevati con relative probabilità
   * - success: indica se la classificazione è andata a buon fine
   * - error: eventuale messaggio di errore
   */
  const [results, setResults] = useState<Array<{
    imageNumber: number;
    imageUrl: string;
    attributes?: Array<{
      name: string;
      probability: number;
      index: number;
      displayName?: string;
    }>;
    success: boolean;
    error?: string;
  }>>([]);
  
  /**
   * isProcessing: Indica se è in corso la classificazione delle immagini
   */
  const [isProcessing, setIsProcessing] = useState(false);
  
  /**
   * currentImage: Numero dell'immagine attualmente in elaborazione
   */
  const [currentImage, setCurrentImage] = useState(0);
  
  /**
   * error: Eventuale messaggio di errore generale
   */
  const [error, setError] = useState<string | null>(null);
  
  /**
   * showDominantOnly: Mostra solo l'attributo dominante per ogni gruppo
   */
  const [showDominantOnly, setShowDominantOnly] = useState(true);
  
  /**
   * showOnlyHighConfidence: Mostra solo attributi con confidenza > 50%
   */
  const [showOnlyHighConfidence, setShowOnlyHighConfidence] = useState(true);
  
  /**
   * modelLoaded: Indica se il modello ONNX è stato caricato con successo
   */
  const [modelLoaded, setModelLoaded] = useState(false);
  
  /**
   * elapsedTime: Tempo totale impiegato per la classificazione
   */
  const [elapsedTime, setElapsedTime] = useState(0);
  
  /**
   * attributeStats: Statistiche aggregate degli attributi dominanti
   * Mappa: nome attributo -> conteggio occorrenze
   */
  const [attributeStats, setAttributeStats] = useState<Record<string, number>>({});

  // ====== CARICAMENTO DEL MODELLO ======
  
  /**
   * useEffect: Carica il modello ONNX all'avvio del componente
   */
  useEffect(() => {
    loadModel();
  }, []);

  /**
   * loadModel: Carica il modello ONNX da file
   */
  const loadModel = async () => {
    try {
      console.log('Loading ONNX model...');
      const modelSession = await ort.InferenceSession.create(CONFIG.modelPath);
      setSession(modelSession);
      setModelLoaded(true);
      console.log('Model loaded successfully');
    } catch (err) {
      const errorMessage = `Failed to load model: ${err instanceof Error ? err.message : String(err)}`;
      setError(errorMessage);
      console.error(errorMessage);
    }
  };

  // ====== PREPROCESSING DELL'IMMAGINE ======
  
  /**
   * preprocessImage: Prepara l'immagine per l'inferenza
   * 
   * Passaggi:
   * 1. Ridimensiona l'immagine a 256x256
   * 2. Normalizza i pixel con mean e std di ImageNet
   * 3. Converte in formato tensor [1, 3, 256, 256] (NCHW)
   * 
   * @param imageElement - Elemento HTML dell'immagine
   * @returns Tensor ONNX pronto per l'inferenza
   */
  const preprocessImage = async (imageElement: HTMLImageElement): Promise<ort.Tensor> => {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      throw new Error('Failed to get canvas context');
    }
    
    // Disegna l'immagine ridimensionata
    ctx.drawImage(imageElement, 0, 0, 256, 256);
    const imageData = ctx.getImageData(0, 0, 256, 256);
    
    // Normalizzazione ImageNet
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    // Alloca array per il tensor [C, H, W]
    const float32Data = new Float32Array(3 * 256 * 256);
    
    // Normalizza e riorganizza i dati (NHWC -> NCHW)
    for (let i = 0; i < imageData.data.length; i += 4) {
      const pixelIndex = i / 4;
      
      // Canale R
      float32Data[pixelIndex] = ((imageData.data[i] / 255) - mean[0]) / std[0];
      // Canale G
      float32Data[256 * 256 + pixelIndex] = ((imageData.data[i + 1] / 255) - mean[1]) / std[1];
      // Canale B
      float32Data[2 * 256 * 256 + pixelIndex] = ((imageData.data[i + 2] / 255) - mean[2]) / std[2];
    }
    
    // Crea tensor ONNX
    return new ort.Tensor('float32', float32Data, [1, 3, 256, 256]);
  };

  // ====== INFERENZA SUL MODELLO ======
  
  /**
   * classifyImage: Esegue la classificazione di una singola immagine
   * 
   * @param imageNumber - Numero identificativo dell'immagine
   * @returns Oggetto con i risultati della classificazione
   */
  const classifyImage = async (imageNumber: number) => {
    if (!session) {
      return {
        imageNumber,
        imageUrl: '',
        success: false,
        error: 'Model not loaded'
      };
    }

    try {
      // Costruisci il percorso dell'immagine (formato: 000001.png)
      const imageUrl = `${CONFIG.imageFolder}/${String(imageNumber).padStart(6, '0')}.png`;
      
      // Carica l'immagine
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = imageUrl;
      });

      // Preprocessing
      const inputTensor = await preprocessImage(img);
      
      // Esegui inferenza
      const feeds = { input: inputTensor };
      const outputData = await session.run(feeds);
      
      // Estrai output (logits raw)
      const output = outputData.output.data as Float32Array;
      
      // Applica sigmoid per ottenere probabilità
      const probabilities = Array.from(output).map(x => 1 / (1 + Math.exp(-x)));
      
      // Crea array di attributi con le loro probabilità
      const attributes = probabilities.map((prob, idx) => ({
        name: CONFIG.attributeNames[idx],
        probability: prob,
        index: idx,
        displayName: CONFIG.displayMapping[idx as keyof typeof CONFIG.displayMapping]
      }));

      return {
        imageNumber,
        imageUrl,
        attributes,
        success: true
      };
    } catch (err) {
      console.error(`Error classifying image ${imageNumber}:`, err);
      return {
        imageNumber,
        imageUrl: `${CONFIG.imageFolder}/${String(imageNumber).padStart(6, '0')}.png`,
        success: false,
        error: err instanceof Error ? err.message : String(err)
      };
    }
  };

  // ====== PROCESSING BATCH ======
  
  /**
   * startClassification: Avvia la classificazione di tutte le immagini
   * Processa le immagini in batch sequenziali e aggiorna le statistiche
   */
  const startClassification = async () => {
    if (!session) {
      setError('Model not loaded');
      return;
    }

    setIsProcessing(true);
    setResults([]);
    setCurrentImage(0);
    setError(null);
    setElapsedTime(0);
    setAttributeStats({});

    const startTime = Date.now();
    const newResults: typeof results = [];
    const stats: Record<string, number> = {};

    // Processa ogni immagine sequenzialmente
    for (let i = 1; i <= CONFIG.targetImages; i++) {
      setCurrentImage(i);
      const result = await classifyImage(i);
      newResults.push(result);

      // Aggiorna statistiche se la classificazione è riuscita
      if (result.success && result.attributes) {
        // Per ogni gruppo di attributi, trova il dominante
        CONFIG.attributeGroups.forEach(group => {
          const groupAttributes = result.attributes!.filter(attr => 
            group.indices.includes(attr.index)
          );
          
          // Trova l'attributo con la probabilità più alta nel gruppo
          const dominant = groupAttributes.reduce((max, attr) => 
            attr.probability > max.probability ? attr : max
          );
          
          // Aggiorna le statistiche
          const displayName = dominant.displayName || dominant.name;
          stats[displayName] = (stats[displayName] || 0) + 1;
        });
        
        // Attributi standalone (smile, facial_hair, eyeglasses)
        const standaloneIndices = [0, 16, 17];
        standaloneIndices.forEach(idx => {
          const attr = result.attributes![idx];
          if (attr.probability > 0.5) {
            const displayName = attr.displayName || attr.name;
            stats[displayName] = (stats[displayName] || 0) + 1;
          }
        });
      }

      // Aggiorna i risultati man mano
      setResults([...newResults]);
    }

    const endTime = Date.now();
    const elapsed = ((endTime - startTime) / 1000).toFixed(2);
    
    setElapsedTime(parseFloat(elapsed));
    setAttributeStats(stats);
    setIsProcessing(false);
  };

  // ====== FILTRAGGIO DEGLI ATTRIBUTI ======
  
  /**
   * getFilteredAttributes: Filtra gli attributi in base alle preferenze dell'utente
   * 
   * @param attributes - Array di attributi da filtrare
   * @returns Array filtrato di attributi
   */
  const getFilteredAttributes = (attributes: Array<{
    name: string;
    probability: number;
    index: number;
    displayName?: string;
  }>) => {
    let filtered = [...attributes];

    // Filtra per confidenza se richiesto
    if (showOnlyHighConfidence) {
      filtered = filtered.filter(attr => attr.probability > 0.5);
    }

    // Mostra solo attributi dominanti se richiesto
    if (showDominantOnly) {
      const dominantAttributes: typeof filtered = [];
      
      // Per ogni gruppo, seleziona l'attributo dominante
      CONFIG.attributeGroups.forEach(group => {
        const groupAttributes = filtered.filter(attr => 
          group.indices.includes(attr.index)
        );
        
        if (groupAttributes.length > 0) {
          const dominant = groupAttributes.reduce((max, attr) => 
            attr.probability > max.probability ? attr : max
          );
          dominantAttributes.push(dominant);
        }
      });
      
      // Aggiungi attributi standalone (smile, facial_hair, eyeglasses)
      const standaloneIndices = [0, 16, 17];
      standaloneIndices.forEach(idx => {
        const attr = filtered.find(a => a.index === idx);
        if (attr && attr.probability > 0.5) {
          dominantAttributes.push(attr);
        }
      });
      
      filtered = dominantAttributes;
    }

    // Ordina per probabilità decrescente
    return filtered.sort((a, b) => b.probability - a.probability);
  };

  // ====== RENDER DEL COMPONENTE ======
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* ====== HEADER ====== */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Facial Attributes Classifier
          </h1>
          <p className="text-gray-600">
            ONNX Runtime Web - ResNet50 Model
          </p>
          <p className="text-sm text-gray-500 mt-1">
            {CONFIG.attributeNames.length} attributes • 256×256 images
          </p>
        </div>

        {/* ====== PANNELLO DI CONTROLLO ====== */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
          <div className="flex flex-wrap items-center gap-4">
            {/* Bottone Start */}
            <button
              onClick={startClassification}
              disabled={!modelLoaded || isProcessing}
              className="flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors shadow-sm hover:shadow"
            >
              <Play size={18} />
              {isProcessing ? 'Processing...' : 'Start Classification'}
            </button>

            {/* Badge stato modello */}
            <div className={`px-3 py-1.5 rounded-full text-sm font-medium ${
              modelLoaded 
                ? 'bg-green-100 text-green-700' 
                : 'bg-gray-100 text-gray-600'
            }`}>
              {modelLoaded ? '● Model Ready' : '○ Loading...'}
            </div>

            {/* Checkbox filtri */}
            {results.length > 0 && (
              <>
                <div className="h-8 w-px bg-gray-300"></div>
                
                {/* Checkbox: Mostra solo dominanti */}
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showDominantOnly}
                    onChange={(e) => setShowDominantOnly(e.target.checked)}
                    className="w-4 h-4 text-indigo-600 rounded focus:ring-2 focus:ring-indigo-500"
                  />
                  <span className="text-sm text-gray-700">Show dominant only</span>
                </label>
                
                {/* Checkbox: Mostra solo alta confidenza (>50%) */}
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
          
          {/* ====== MESSAGGI DI ERRORE ====== */}
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm mt-4">
              <strong>Error:</strong> {error}
            </div>
          )}
          
          {/* ====== BARRA DI PROGRESSO ====== */}
          {isProcessing && (
            <div className="bg-blue-50 border border-blue-200 px-4 py-3 rounded-lg mt-4">
              <div className="flex items-center gap-3 mb-2">
                {/* Spinner animato */}
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <span className="text-blue-700 text-sm font-medium">
                  Processing: {currentImage}/{CONFIG.targetImages}
                </span>
              </div>
              {/* Barra di progresso visuale */}
              <div className="bg-blue-200 rounded-full h-1.5 overflow-hidden">
                <div 
                  className="bg-blue-600 h-full transition-all duration-300"
                  style={{ width: `${(currentImage / CONFIG.targetImages) * 100}%` }}
                ></div>
              </div>
            </div>
          )}

          {/* ====== MESSAGGIO DI COMPLETAMENTO ====== */}
          {elapsedTime > 0 && !isProcessing && (
            <div className="bg-green-50 border border-green-200 px-4 py-3 rounded-lg mt-4">
              <div className="flex items-center gap-2">
                <CheckCircle size={18} className="text-green-600" />
                <span className="text-green-700 text-sm font-medium">
                  Completed in {elapsedTime} seconds
                </span>
              </div>
            </div>
          )}
        </div>

        {/* ====== SEZIONE STATISTICHE ATTRIBUTI ====== */}
        {Object.keys(attributeStats).length > 0 && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Statistiche Attributi Dominanti</h2>
            <div className="bg-gray-50 rounded-lg p-4 font-mono text-sm">
              <pre className="whitespace-pre-wrap">
                {Object.entries(attributeStats as Record<string, number>)
                  .sort((a, b) => b[1] - a[1])
                  .map(([attribute, count], index) => {
                    const percentage = ((count / CONFIG.targetImages) * 100).toFixed(0);
                    const position = String(index + 1).padStart(2, ' ');
                    const attributePadded = attribute.padEnd(25, ' ');

                    // QUESTI SONO I DATI IMPORTANTI CHE VENGONO USATI DAL BOT
                    console.log(`${position}. ${attributePadded} ${count} (${percentage}%)`);

                    return `${position}. ${attributePadded} ${count} (${percentage}%)`;
                  })
                  .join('\n')}
              </pre>
            </div>
          </div>
        )}
        
        {/* ====== GRIGLIA DEI RISULTATI ====== */}
        {results.length > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {results.map((result) => {
              const filteredAttributes = getFilteredAttributes(result.attributes || []);
              return (
                <div key={result.imageNumber} className="bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow overflow-hidden">
                  {/* ====== IMMAGINE ====== */}
                  <div className="relative">
                    <img 
                      src={result.imageUrl} 
                      alt={`${result.imageNumber}`}
                      className="w-full aspect-square object-cover"
                      onError={(e) => {
                        const target = e.target as HTMLElement;
                        target.style.display = 'none';
                        (target.nextSibling as HTMLElement).style.display = 'flex';
                      }}
                    />
                    {/* Placeholder in caso di errore */}
                    <div className="hidden w-full aspect-square bg-gray-100 items-center justify-center text-gray-400 flex-col gap-2">
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                        <polyline points="21 15 16 10 5 21"></polyline>
                      </svg>
                      <span className="text-xs">{String(result.imageNumber).padStart(6, '0')}.png</span>
                    </div>
                    {/* Badge con il numero dell'immagine */}
                    <div className="absolute top-1.5 right-1.5 bg-white/90 backdrop-blur-sm px-2 py-0.5 rounded text-xs font-semibold text-gray-700">
                      {String(result.imageNumber).padStart(6, '0')}
                    </div>
                  </div>
                  
                  {/* ====== LISTA ATTRIBUTI ====== */}
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
        
        {/* ====== STATO INIZIALE ====== */}
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