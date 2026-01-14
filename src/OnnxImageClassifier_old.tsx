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
  modelPath: 'facial_attributes_model.onnx',
  
  // Numero totale di immagini da processare (24 per il gioco)
  targetImages: 24,
  
  // Cartella contenente le immagini da analizzare (risoluzione 224x224)
  imageFolder: 'images_224',
  
  /**
   * NOMI DEGLI ATTRIBUTI
   * Array di 39 descrizioni testuali per ciascun attributo che il modello può rilevare.
   * L'ordine corrisponde agli output del modello ONNX.
   */
  attributeNames: [
    // Genere (indici 0-3)
    'a man',   //'Gender'
    'a woman',
    'a boy',
    'a girl',

    // Colore capelli (indici 4-10)
    'a person with blond hair',  //'Hair Color'
    'a person with brown hair',
    'a person with black hair',
    'a person with red hair',
    'a person with gray hair',
    'a person with white hair',
    'a bald person',

    // Lunghezza capelli (indici 11-13)
    'a bald person',           //'Hair Length'
    'a person with short hair',
    'a person with hair around his neck',

    // Tipo di capelli (indici 14-18)
    'a bald person',              //'Hair Type'
    'a person with straight hair',
    'a person with curly hair',
    'a person with wavy hair',
    'a person with afro hair',

    // Peli facciali (indici 19-21)
    'clean shaven',     //'Facial Hair'
    'stubble',
    'beard',

    // Colore occhi (indici 22-27)
    'a person with brown eyes', //'Eye Color'
    'a person with blue eyes',
    'a person with green eyes',
    'a person with hazel eyes',
    'a person with black eyes',
    'a person with amber eyes',

    // Cappello/copricapo (indici 28-33)
    'a person with clean hair', //'Hat'
    'a person with few hair',
    'a woman with her head covered',
    'a woman with visible forehead',
    'a woman with a headband on her forehead',
    'a person with a hat',

    // Occhiali e caratteristiche occhi (indici 34-38)
    'a person with eyes',  //'Eyeglasses'
    'a person with visible eyes',
    'a person with eye wrinkles',
    'a person with eye bags',
    'a person with eyeglasses'
  ],
  
  /**
   * GRUPPI DI ATTRIBUTI MUTUALMENTE ESCLUSIVI
   * Definisce quali attributi appartengono alla stessa categoria.
   * In ogni gruppo, solo l'attributo con la probabilità più alta viene considerato dominante.
   */
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
  
  /**
   * MAPPATURA PER LA VISUALIZZAZIONE
   * Converte gli indici degli attributi in etichette leggibili in italiano.
   * Utilizzato quando showDominantOnly è attivo.
   */
  displayMapping: {
    // Genere (0-3)
    0: 'Uomo',
    1: 'Donna',
    2: 'Uomo',
    3: 'Donna',
    
    // Colore capelli (4-10)
    4: 'Capelli biondi',
    5: 'Capelli Marroni',
    6: 'Capelli neri',
    7: 'Capelli rossi',
    8: 'Capelli grigi',
    9: 'Capelli bianchi',
    10: 'No Capelli',
    
    // Lunghezza capelli (11-13)
    11: 'No capelli',
    12: 'Capelli corti',
    13: 'Capelli lunghi',
    
    // Tipo capelli (14-18)
    14: 'No capelli',
    15: 'Capelli lisci',
    16: 'Capelli ricci',
    17: 'Capelli mossi',//questi due possono rientrare in ricci
    18: 'Capelli afro',
    
    // Peli facciali (19-21)
    19: 'Senza Barba', 
    20: 'Con Barba',
    21: 'Con Barba',
    
    // Colore occhi (22-27)
    22: 'Occhi marroni', //'Brown Eyes',
    23: 'Occhi azzurri',
    24: 'Occhi verdi',
    25: 'Occhi verdi',
    26: 'Occhi marroni',
    27: 'Occhi verdi',
    
    // Cappello (28-33)
    28: 'Senza Cappello',
    29: 'Senza Cappello',
    30: 'Con Cappello',
    31: 'Senza Cappello',
    32: 'Con Cappello',
    33: 'Con Cappello',
    
    // Occhiali (34-38)
    34: 'Senza Occhiali',
    35: 'Senza Occhiali',
    36: 'Senza Occhiali',
    37: 'Senza Occhiali',
    38: 'Con Occhiali'
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
    attributes?: Array<{ name: string | undefined; probability: number; rawValue: number; index: number; displayName?: string }>;
    success: boolean;
    error?: string;
  }>>([]);
  
  /**
   * currentImage: Numero dell'immagine attualmente in elaborazione
   * Utilizzato per mostrare il progresso durante il batch processing
   */
  const [currentImage, setCurrentImage] = useState(0);
  
  /**
   * isProcessing: Flag che indica se è in corso un'elaborazione batch
   */
  const [isProcessing, setIsProcessing] = useState(false);
  
  /**
   * error: Messaggio di errore globale (es. errore nel caricamento del modello)
   */
  const [error, setError] = useState<string | null>(null);
  
  /**
   * modelLoaded: Flag che indica se il modello ONNX è stato caricato correttamente
   */
  const [modelLoaded, setModelLoaded] = useState(false);
  
  /**
   * showOnlyHighConfidence: Filtro per mostrare solo attributi con probabilità > 50%
   */
  const [showOnlyHighConfidence, setShowOnlyHighConfidence] = useState(false);
  
  /**
   * elapsedTime: Tempo impiegato per processare tutte le immagini (in secondi)
   */
  const [elapsedTime, setElapsedTime] = useState(0);
  
  /**
   * showDominantOnly: Se true, mostra solo l'attributo dominante per ogni gruppo
   * (es. solo il colore di capelli più probabile invece di tutti i colori)
   */
  const [showDominantOnly, setShowDominantOnly] = useState(true);
  
  /**
   * attributeStats: Oggetto contenente statistiche aggregate degli attributi dominanti
   * Chiave: nome dell'attributo in italiano
   * Valore: numero di immagini in cui quell'attributo è stato rilevato come dominante
   */
  const [attributeStats, setAttributeStats] = useState({});

  /**
   * FUNZIONE: loadModel
   * 
   * Carica il modello ONNX e i suoi dati esterni dal server.
   * Il modello è suddiviso in due file:
   * - facial_attributes_model.onnx: struttura del modello
   * - facial_attributes_model.onnx.data: pesi del modello
   * 
   * Questa suddivisione è comune per modelli di grandi dimensioni.
   */
  const loadModel = async () => {
    try {
      // Reset dell'eventuale errore precedente
      setError(null);
      
      // Carica il file principale del modello
      const modelResponse = await fetch('/facial_attributes_model.onnx');
      const modelBuffer = await modelResponse.arrayBuffer();
      
      // Carica i dati esterni (pesi del modello)
      const dataResponse = await fetch('/facial_attributes_model.onnx.data');
      const dataBuffer = await dataResponse.arrayBuffer();
      
      // Crea la sessione ONNX Runtime specificando i dati esterni
      const sess = await ort.InferenceSession.create(modelBuffer, {
        externalData: [
          {
            data: dataBuffer,
            path: 'facial_attributes_model.onnx.data'
          }
        ]
      });
      
      // Aggiorna lo stato per indicare che il modello è pronto
      setSession(sess);
      setModelLoaded(true);
      console.log('Modello caricato con successo');
    } catch (err) {
      // Gestione degli errori durante il caricamento
      const message = err instanceof Error ? err.message : String(err);
      setError(`Errore nel caricamento del modello: ${message}`);
      console.error(err);
    }
  };
  
  /**
   * FUNZIONE: preprocessImage
   * 
   * Pre-processa un'immagine per renderla compatibile con l'input del modello ONNX.
   * 
   * Passaggi:
   * 1. Carica l'immagine e ridimensionala a 224x224 pixel
   * 2. Estrae i valori RGB pixel per pixel
   * 3. Normalizza i valori da [0-255] a [0-1]
   * 4. Organizza i dati in formato NCHW (batch, canali, altezza, larghezza)
   *    - N=1 (una sola immagine)
   *    - C=3 (RGB)
   *    - H=224 (altezza)
   *    - W=224 (larghezza)
   * 5. Crea un tensore ONNX con shape [1, 3, 224, 224]
   * 
   * @param imageUrl - URL dell'immagine da pre-processare
   * @returns Promise che risolve in un tensore ONNX pronto per l'inferenza
   */
  const preprocessImage = async (imageUrl: string): Promise<ort.Tensor> => {
    return new Promise<ort.Tensor>((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous'; // Abilita CORS per immagini da altri domini
      
      img.onload = () => {
        try {
          // Crea un canvas HTML5 per manipolare l'immagine
          const canvas = document.createElement('canvas');
          canvas.width = 224;
          canvas.height = 224;
          const ctx = canvas.getContext('2d');

          if (!ctx) {
            reject(new Error('Unable to obtain 2D rendering context'));
            return;
          }
          
          // Disegna l'immagine ridimensionata sul canvas
          ctx.drawImage(img, 0, 0, 224, 224);
          
          // Estrai i dati dei pixel come array RGBA
          const imageData = ctx.getImageData(0, 0, 224, 224);
          const data = imageData.data;
          
          // Array per separare i canali RGB
          const red: number[] = [], green: number[] = [], blue: number[] = [];
          
          // Itera sui pixel (ogni pixel ha 4 valori: R, G, B, A)
          for (let i = 0; i < data.length; i += 4) {
            const r = data[i] ?? 0;      // Rosso
            const g = data[i + 1] ?? 0;  // Verde
            const b = data[i + 2] ?? 0;  // Blu
            // L'alpha (data[i + 3]) viene ignorato
            
            // Normalizza i valori da [0-255] a [0-1]
            red.push(r / 255.0);
            green.push(g / 255.0);
            blue.push(b / 255.0);
          }
          
          // Concatena i canali nell'ordine RGB (formato richiesto dal modello)
          const input = new Float32Array([...red, ...green, ...blue]);
          
          // Crea il tensore ONNX con shape [1, 3, 224, 224]
          const tensor = new ort.Tensor('float32', input, [1, 3, 224, 224]);
          
          resolve(tensor);
        } catch (err) {
          reject(err instanceof Error ? err : new Error(String(err)));
        }
      };
      
      // Gestione errori nel caricamento dell'immagine
      img.onerror = () => reject(new Error(`Impossibile caricare l'immagine: ${imageUrl}`));
      img.src = imageUrl;
    });
  };

  /**
   * FUNZIONE: classifyImage
   * 
   * Classifica una singola immagine usando il modello ONNX caricato.
   * 
   * Processo:
   * 1. Costruisce il percorso dell'immagine con zero-padding (es. 000001.png)
   * 2. Pre-processa l'immagine in un tensore
   * 3. Esegue l'inferenza con il modello ONNX
   * 4. Applica la funzione sigmoide agli output per ottenere probabilità [0-1]
   * 5. Costruisce l'array di attributi con nomi e probabilità
   * 
   * @param imageNumber - Numero progressivo dell'immagine (1-24)
   * @param sess - Sessione ONNX Runtime da utilizzare
   * @returns Oggetto contenente i risultati della classificazione
   */
  const classifyImage = async (imageNumber: number, sess: ort.InferenceSession) => {
    try {
      // Formatta il numero con zero-padding (es. 1 -> "000001")
      const paddedNumber = String(imageNumber).padStart(6, '0');
      const imageUrl: string = `${CONFIG.imageFolder}/${paddedNumber}.png`;
      
      // Pre-processa l'immagine
      const tensor: ort.Tensor = await preprocessImage(imageUrl);
      
      // Prepara l'input per il modello
      const feeds: Record<string, ort.Tensor> = {};
      const inputName: string | undefined = sess.inputNames && sess.inputNames.length > 0 ? sess.inputNames[0] : 'input';
      if(inputName != undefined){
        feeds[inputName] = tensor;
      }
      
      // Esegue l'inferenza
      const output: ort.InferenceSession.OnnxValueMapType = await sess.run(feeds);
      
      // Recupera il nome dell'output
      const outputName: string | undefined = sess.outputNames && sess.outputNames.length > 0 ? sess.outputNames[0] : Object.keys(output)[0];
      
      if(outputName === undefined) {
        throw new Error('Unable to determine output name');
      }
      
      const outputTensor = output[outputName];
      if(!outputTensor) {
        throw new Error('Output tensor is undefined');
      }
      
      // Estrae le predizioni grezze (logits)
      const predictions = outputTensor.data as ArrayLike<number>;
      
      // Converte ogni predizione in probabilità usando la funzione sigmoide
      // e costruisce l'array di attributi
      const allAttributes = [];
      for (let i = 0; i < predictions.length && i < CONFIG.attributeNames.length; i++) {
        const pred = predictions[i];
        if(pred !== undefined && pred !== null){
          // Funzione sigmoide: 1 / (1 + e^(-x))
          // Converte logits in probabilità [0-1]
          const probability = 1 / (1 + Math.exp(-pred));
          allAttributes.push({
            name: CONFIG.attributeNames[i],
            probability: probability,
            rawValue: pred,  // Valore grezzo (logit)
            index: i         // Indice dell'attributo
          });
        }
      }

      // Ritorna il risultato della classificazione
      return {
        imageNumber,
        imageUrl,
        attributes: allAttributes,
        success: true
      };
    } catch (err) {
      // Gestione errori durante la classificazione
      const message = err instanceof Error ? err.message : String(err);
      return {
        imageNumber,
        imageUrl: `${CONFIG.imageFolder}/${String(imageNumber).padStart(6, '0')}.png`,
        error: message,
        success: false
      };
    }
  };

  /**
   * FUNZIONE: processAllImages
   * 
   * Processa in sequenza tutte le immagini configurate (1 a CONFIG.targetImages).
   * 
   * Processo:
   * 1. Valida che il modello sia caricato
   * 2. Inizializza gli stati per il tracciamento del progresso
   * 3. Itera su tutte le immagini, classificandole una alla volta
   * 4. Aggiorna lo stato dei risultati dopo ogni immagine
   * 5. Calcola le statistiche degli attributi dominanti
   * 6. Registra il tempo totale di elaborazione
   */
  const processAllImages = async () => {
    // Validazione: il modello deve essere caricato
    if (!session) {
      setError('Carica prima il modello');
      return;
    }
    
    const sess = session as ort.InferenceSession;
    
    // Reset degli stati prima di iniziare
    setIsProcessing(true);
    setResults([]);
    setError(null);
    setElapsedTime(0);
    setAttributeStats({});
    
    // Inizia il timer
    const start = Date.now();
    const allResults = [];
    
    // Processa ogni immagine sequenzialmente
    for (let i = 1; i <= CONFIG.targetImages; i++) {
      setCurrentImage(i);  // Aggiorna il contatore del progresso
      const result = await classifyImage(i, sess);
      allResults.push(result);
      setResults(prev => [...prev, result]);  // Aggiunge il risultato alla lista
    }
    
    /**
     * CALCOLO STATISTICHE DEGLI ATTRIBUTI DOMINANTI
     * 
     * Per ogni immagine:
     * 1. Identifica gli attributi dominanti (uno per gruppo)
     * 2. Conta quante volte ogni attributo appare come dominante
     * 3. Usa un Set per contare ogni immagine una sola volta per attributo
     */
    const stats: Record<string, Set<number>> = {};
    allResults.forEach(result => {
      if (result.success && result.attributes!==undefined) {
        // Ottiene gli attributi dominanti per questa immagine
        const dominantAttrs = getDominantAttributes(result.attributes);
        dominantAttrs.forEach(attr => {
          const key: string | undefined = attr.displayName || attr.name;
          // Conta solo una volta per immagine anche se appare più volte
          if(key!== undefined){
            if (!stats[key]) {
              stats[key] = new Set();
            }
            // Aggiunge il numero dell'immagine al Set (evita duplicati)
            stats[key].add(result.imageNumber);
          }
        });
      }
    });
    
    // Converte i Set in conteggi numerici
    const finalStats: Record<string, number> = {};
    Object.keys(stats).forEach(key => {
      if(stats[key]!==undefined)
      finalStats[key] = stats[key].size;  // .size restituisce il numero di elementi unici
    });
    
    setAttributeStats(finalStats);
    
    // Calcola e registra il tempo totale di elaborazione
    const totalTime: SetStateAction<number> = Number(((Date.now() - start) / 1000).toFixed(2));
    setElapsedTime(totalTime);
    setIsProcessing(false);
    setCurrentImage(0);
  };

/**
 * TIPO: Attribute  (aggiunto per non far rompere il cazzo a typescript)
 * 
 * Definisce la struttura di un attributo facciale rilevato:
 * - name: descrizione testuale dell'attributo (es. "a person with blond hair")
 * - probability: probabilità [0-1] che l'attributo sia presente
 * - rawValue: valore grezzo (logit) prima della sigmoide
 * - index: posizione nell'array degli attributi
 * - displayName: nome in italiano per la visualizzazione (opzionale)
 */
type Attribute = {
  name: string | undefined;
  probability: number;
  rawValue: number;
  index: number;
  displayName?: string;
};

/**
 * FUNZIONE: getDominantAttributes
 * 
 * Seleziona l'attributo dominante (con probabilità più alta) da ogni gruppo mutualmente esclusivo.
 * 
 * Esempio: Se un gruppo contiene "capelli biondi", "capelli castani", "capelli neri",
 * questa funzione seleziona solo quello con la probabilità più alta.
 * 
 * Processo:
 * 1. Per ogni gruppo definito in CONFIG.attributeGroups:
 *    a. Trova l'attributo con la probabilità massima
 *    b. Applica il displayName se disponibile
 *    c. Aggiunge l'attributo ai risultati
 *    d. Marca tutti gli indici del gruppo come "usati"
 * 2. Aggiunge eventuali attributi che non appartengono a nessun gruppo
 * 
 * @param attributes - Array completo di attributi da filtrare
 * @returns Array contenente solo gli attributi dominanti
 */
const getDominantAttributes = (attributes: Attribute[]): Attribute[] => {
  const dominantAttrs: Attribute[] = [];
  const usedIndices = new Set<number>();  // Tiene traccia degli indici già processati
  
  // Itera su ogni gruppo di attributi mutualmente esclusivi
  for (const group of CONFIG.attributeGroups) {
    let maxProb = -1;
    let maxAttr: Attribute | undefined = undefined;
    
    // Trova l'attributo con la probabilità massima nel gruppo
    for (const idx of group.indices) {
      const attr = attributes.find((a: Attribute) => a.index === idx);
      if (attr && attr.probability > maxProb) {
        maxProb = attr.probability;
        maxAttr = attr;
      }
    }
    
    // Se trovato un attributo dominante, aggiungilo ai risultati
    if (maxAttr) {
      // Cerca il nome italiano corrispondente
      const displayName = CONFIG.displayMapping[maxAttr.index as keyof typeof CONFIG.displayMapping] || maxAttr.name;
      
      if (displayName) {
        // Crea un nuovo oggetto con il displayName
        dominantAttrs.push({
          name: maxAttr.name,
          probability: maxAttr.probability,
          rawValue: maxAttr.rawValue,
          index: maxAttr.index,
          displayName: displayName
        });
      } else {
        // Se non c'è displayName, usa l'attributo originale
        dominantAttrs.push({
          name: maxAttr.name,
          probability: maxAttr.probability,
          rawValue: maxAttr.rawValue,
          index: maxAttr.index
        });
      }
      
      // Marca tutti gli indici di questo gruppo come usati
      for (const idx of group.indices) {
        usedIndices.add(idx);
      }
    }
  }
  
  // Aggiungi attributi che non appartengono a nessun gruppo
  for (const attr of attributes) {
    if (!usedIndices.has(attr.index)) {
      const displayName = CONFIG.displayMapping[attr.index as keyof typeof CONFIG.displayMapping] || attr.name;
      
      if (displayName) {
        dominantAttrs.push({
          ...attr,
          displayName: displayName
        });
      } else {
        dominantAttrs.push(attr);
      }
    }
  }
  
  return dominantAttrs;
};

  /**
   * FUNZIONE: getFilteredAttributes
   * 
   * Applica i filtri di visualizzazione agli attributi in base alle preferenze dell'utente:
   * 1. Se showDominantOnly è attivo, mostra solo l'attributo dominante per gruppo
   * 2. Se showOnlyHighConfidence è attivo, mostra solo attributi con probabilità > 50%
   * 
   * @param attributes - Array di attributi da filtrare
   * @returns Array di attributi filtrati secondo le preferenze correnti
   */
  const getFilteredAttributes = (attributes: Array<{ name: string | undefined; probability: number; rawValue: number; index: number; displayName?: string }>) => {
    let filtered = attributes;
    
    // Primo filtro: mostra solo attributi dominanti
    if (showDominantOnly) {
      filtered = getDominantAttributes(filtered);
    }
    
    // Secondo filtro: mostra solo alta confidenza (>50%)
    if (showOnlyHighConfidence) {
      filtered = filtered.filter(attr => attr.probability > 0.5);
    }
    
    return filtered;
  };

  /**
   * RENDER DEL COMPONENTE
   * 
   * Struttura dell'interfaccia:
   * 1. Header con titolo e controlli principali
   * 2. Area di stato (errori, progresso, completamento)
   * 3. Statistiche degli attributi dominanti (se disponibili)
   * 4. Griglia di risultati con immagini e attributi classificati
   */
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* ====== SEZIONE HEADER E CONTROLLI ====== */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">
            Facial Attributes Classifier
          </h1>
          
          {/* Pulsanti di controllo e checkbox */}
          <div className="flex flex-wrap items-center gap-3 mb-4">
            {/* Pulsante per caricare il modello */}
            <button
              onClick={loadModel}
              disabled={modelLoaded || isProcessing}
              className="px-5 py-2.5 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              {modelLoaded ? '✓ Model Loaded' : 'Load Model'}
            </button>
            
            {/* Pulsante per avviare la classificazione */}
            <button
              onClick={processAllImages}
              disabled={!modelLoaded || isProcessing}
              className="px-5 py-2.5 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              <Play size={18} />
              {isProcessing ? `Processing ${currentImage}/${CONFIG.targetImages}...` : 'Start Classification'}
            </button>

            {/* Checkbox per i filtri (visibili solo dopo aver ottenuto risultati) */}
            {results.length > 0 && (
              <>
                {/* Checkbox: Mostra solo attributi dominanti */}
                <label className="flex items-center gap-2 cursor-pointer ml-4">
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
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
              <strong>Error:</strong> {error}
            </div>
          )}
          
          {/* ====== BARRA DI PROGRESSO ====== */}
          {isProcessing && (
            <div className="bg-blue-50 border border-blue-200 px-4 py-3 rounded-lg">
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

        {/* ====== SEZIONE STATISTICHE ATTRIBUTI ====== */}
        {/* Mostra le statistiche aggregate solo se disponibili */}
        {Object.keys(attributeStats).length > 0 && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Statistiche Attributi Dominanti</h2>
            <div className="bg-gray-50 rounded-lg p-4 font-mono text-sm">
              <pre className="whitespace-pre-wrap">
                {/* 
                  PARTE IMPORTANTE
                  Formatta le statistiche come lista ordinata:
                  - Ordina per conteggio decrescente
                  - Mostra numero, nome attributo, conteggio e percentuale
                  - Esempio: " 1. Capelli neri          18 (75%)"
                */}
                {Object.entries(attributeStats as Record<string, number>)
                  .sort((a, b) => b[1] - a[1])  // Ordina per conteggio decrescente
                  .map(([attribute, count], index) => {
                    const percentage = ((count / CONFIG.targetImages) * 100).toFixed(0);
                    const position = String(index + 1).padStart(2, ' ');
                    const attributePadded = attribute.padEnd(25, ' ');



                    //QUESTI SONO I DATI IMPORTANTI CHE VENGONO USATI DAL BOT
                    console.log(`${position}. ${attributePadded} ${count} (${percentage}%)`)
                    //console.log(Object)


                    return `${position}. ${attributePadded} ${count} (${percentage}%)`;
                  })
                  .join('\n')}
                  
              </pre>
            </div>
          </div>
        )}
        
        {/* ====== GRIGLIA DEI RISULTATI ====== */}
        {/* Mostra i risultati solo se disponibili */}
        {results.length > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {results.map((result) => {
              // Applica i filtri agli attributi prima di visualizzarli
              const filteredAttributes = getFilteredAttributes(result.attributes || []);
              return (
                <div key={result.imageNumber} className="bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow overflow-hidden">
                  {/* ====== IMMAGINE ====== */}
                  <div className="relative">
                    {/* Immagine principale */}
                    <img 
                      src={result.imageUrl} 
                      alt={`${result.imageNumber}`}
                      className="w-full aspect-square object-cover"
                      onError={(e) => {
                        // Gestione errore caricamento immagine: mostra placeholder
                        const target = e.target as HTMLElement;
                        target.style.display = 'none';
                        (target.nextSibling as HTMLElement).style.display = 'flex';
                      }}
                    />
                    {/* Placeholder in caso di errore */}
                    <div className="hidden w-full aspect-square bg-gray-100 items-center justify-center text-gray-400 flex-col gap-2">
                      {/* Icona immagine mancante */}
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
                        {/* Mappa ogni attributo filtrato */}
                        {filteredAttributes.map((attr, idx) => (
                          <div key={idx} className="flex items-center justify-between gap-2 text-xs">
                            {/* Nome dell'attributo (italiano se disponibile) */}
                            <span className="text-gray-700 truncate flex-1">
                              {showDominantOnly && attr.displayName ? attr.displayName : attr.name}
                            </span>
                            {/* Percentuale con colore in base alla confidenza */}
                            <span className={`font-semibold flex-shrink-0 ml-2 ${
                              attr.probability > 0.7 ? 'text-green-600' :   // Alta confidenza (>70%)
                              attr.probability > 0.5 ? 'text-blue-600' :    // Media confidenza (50-70%)
                              'text-gray-500'                               // Bassa confidenza (<50%)
                            }`}>
                              {(attr.probability * 100).toFixed(0)}%
                            </span>
                          </div>
                        ))}
                        {/* Messaggio se nessun attributo passa i filtri */}
                        {filteredAttributes.length === 0 && (
                          <p className="text-xs text-gray-400 text-center py-2">No attributes match filters</p>
                        )}
                      </div>
                    ) : (
                      // Messaggio di errore se la classificazione è fallita
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
        
        {/* ====== STATO INIZIALE (MODELLO CARICATO, NESSUN RISULTATO) ====== */}
        {!isProcessing && results.length === 0 && modelLoaded && (
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-12 text-center">
            {/* Icona placeholder */}
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