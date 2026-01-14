import * as ort from 'onnxruntime-web';

/**
 * CONFIGURAZIONE
 */
const CONFIG = {
  modelPath: 'model_webgpu.onnx',
  imageFolder: 'images_256',
  attributeNames: [
    'smile',
    'gender_male',
    'gender_female',
    'hair_brown',
    'hair_black',
    'hair_blond',
    'hair_gray',
    'hair_long',
    'hair_short',
    'ethnicity_asian',
    'ethnicity_black',
    'ethnicity_latino',
    'ethnicity_white',
    'eye_blue',
    'eye_brown',
    'eye_green',
    'has_facial_hair',
    'a person with eyeglasses'
  ],
  attributeGroups: [
    { name: 'Gender', indices: [1, 2] },
    { name: 'Hair Color', indices: [3, 4, 5, 6] },
    { name: 'Hair Length', indices: [7, 8] },
    { name: 'Ethnicity', indices: [9, 10, 11, 12] },
    { name: 'Eye Color', indices: [13, 14, 15] }
  ],
  displayMapping: {
    0: 'Sorriso',
    1: 'Uomo',
    2: 'Donna',
    3: 'Capelli Marroni',
    4: 'Capelli Neri',
    5: 'Capelli Biondi',
    6: 'Capelli Grigi',
    7: 'Capelli Lunghi',
    8: 'Capelli Corti',
    9: 'Asiatico',
    10: 'Nero',
    11: 'Latino',
    12: 'Bianco',
    13: 'Occhi Azzurri',
    14: 'Occhi Marroni',
    15: 'Occhi Verdi',
    16: 'Con Barba',
    17: 'Con Occhiali'
  }
};

// Mapping domanda -> tratto -> indici attributi originali
const QUESTION_TO_INDICES: Record<number, number[]> = {
  0: [0],              // Sorriso (aggiunto dal nuovo modello)
  1: [1],              // Uomo
  2: [2],              // Donna
  3: [3],              // Capelli Marroni
  4: [4],              // Capelli neri
  5: [5],              // Capelli biondi
  6: [6],              // Capelli grigi (come "capelli rossi" nel vecchio)   PELATO DA AGGIUNGERE IN CASO IN SEGUITO
  7: [7],              // Capelli lunghi
  8: [8],              // Capelli corti
  9: [9],              // Etnia asiatico
  10: [10],            // Etnia africano
  11: [11],            // Etnia latino
  12: [12],            // Etnia caucasica
  13: [13],            // Occhi azzurri
  14: [14],            // Occhi marroni
  15: [15],            // Occhi verdi
  16: [16],           // Con Barba
  17: [17],            // Con Occhiali          CAPPELLO DA AGGIUNGERE IN CASO IN SEGUITO
    //Domande sulle lettere non presenti qui 

};

/**
 * TIPI
 */
type Attribute = {
  name: string | undefined;
  probability: number;
  rawValue: number;
  index: number;
  displayName?: string;
};

export type QuestionAnswer = {
  questionId: number;
  answer: boolean;
  percentage: number;
};

/**
 * CLASSE PRINCIPALE CON SUPPORTO WEBGPU
 */
export class FacialAttributesClassifier {
  private session: ort.InferenceSession | null = null;
  private executionProvider: string = '';
  
  /**
   * Carica il modello con supporto WebGPU
   * @param useWebGpu - Se true, tenta di usare WebGPU, altrimenti usa WASM
   * @param modelPath - Percorso personalizzato del modello (opzionale)
   */
  async loadModel(useWebGpu: boolean = true, modelPath?: string): Promise<void> {
    const mPath = modelPath || CONFIG.modelPath;
    
    try {
      // Configurazione delle opzioni di sessione
      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: [],
        graphOptimizationLevel: 'all',
      };

      // Determina quale execution provider utilizzare
      if (useWebGpu) {
        // Verifica supporto WebGPU
        if ('gpu' in navigator) {
          try {
            const adapter = await (navigator as any).gpu.requestAdapter();
            if (adapter) {
              console.log('WebGPU supported, attempting to use it...');
              sessionOptions.executionProviders = ['webgpu'];
            } else {
              console.warn('WebGPU not available, falling back to WASM');
              sessionOptions.executionProviders = ['wasm'];
            }
          } catch (err) {
            console.warn('WebGPU check failed, using WASM:', err);
            sessionOptions.executionProviders = ['wasm'];
          }
        } else {
          console.warn('WebGPU not supported by browser, using WASM');
          sessionOptions.executionProviders = ['wasm'];
        }
      } else {
        console.log('Using WASM (CPU) as requested');
        sessionOptions.executionProviders = ['wasm'];
      }

      // Crea la sessione
      this.session = await ort.InferenceSession.create(mPath, sessionOptions);
      
      // Verifica quale provider è stato effettivamente utilizzato
      this.executionProvider = (this.session as any).handler?._ep || 
                              sessionOptions.executionProviders[0] || 
                              'unknown';
      
      console.log('Model loaded successfully');
      console.log('Execution provider:', this.executionProvider);
      
    } catch (error) {
      throw new Error(`Failed to load model: ${error}`);
    }
  }
  
  /**
   * Pre-processa un'immagine 256x256 per il modello
   */
  private async preprocessImage(imageUrl: string): Promise<ort.Tensor> {
    return new Promise<ort.Tensor>((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas');
          canvas.width = 256;
          canvas.height = 256;
          const ctx = canvas.getContext('2d');
          
          if (!ctx) {
            reject(new Error('Unable to obtain 2D rendering context'));
            return;
          }
          
          ctx.drawImage(img, 0, 0, 256, 256);
          const imageData = ctx.getImageData(0, 0, 256, 256);
          const data = imageData.data;
          
          const red: number[] = [], green: number[] = [], blue: number[] = [];
          
          for (let i = 0; i < data.length; i += 4) {
            const r = data[i] ?? 0;
            const g = data[i + 1] ?? 0;
            const b = data[i + 2] ?? 0;
            
            red.push(r / 255.0);
            green.push(g / 255.0);
            blue.push(b / 255.0);
          }
          
          const input = new Float32Array([...red, ...green, ...blue]);
          const tensor = new ort.Tensor('float32', input, [1, 3, 256, 256]);
          
          resolve(tensor);
        } catch (err) {
          reject(err instanceof Error ? err : new Error(String(err)));
        }
      };
      
      img.onerror = () => reject(new Error(`Failed to load image: ${imageUrl}`));
      img.src = imageUrl;
    });
  }
  
  /**
   * Ottiene gli attributi dominanti per ogni gruppo
   */
  private getDominantAttributes(attributes: Attribute[]): Map<string, number> {
    const dominantMap = new Map<string, number>();
    
    // Processa i gruppi di attributi mutualmente esclusivi
    for (const group of CONFIG.attributeGroups) {
      let maxProb = -1;
      let maxAttr: Attribute | undefined = undefined;
      
      for (const idx of group.indices) {
        const attr = attributes.find(a => a.index === idx);
        if (attr && attr.probability > maxProb) {
          maxProb = attr.probability;
          maxAttr = attr;
        }
      }
      
      if (maxAttr) {
        const displayName = CONFIG.displayMapping[maxAttr.index as keyof typeof CONFIG.displayMapping];
        if (displayName) {
          dominantMap.set(displayName, maxAttr.probability);
        }
      }
    }
    
    // Aggiungi attributi non in gruppi (smile, facial hair, eyeglasses)
    const usedIndices = new Set<number>();
    CONFIG.attributeGroups.forEach(g => g.indices.forEach(i => usedIndices.add(i)));
    
    for (const attr of attributes) {
      if (!usedIndices.has(attr.index)) {
        const displayName = CONFIG.displayMapping[attr.index as keyof typeof CONFIG.displayMapping];
        if (displayName) {
          dominantMap.set(displayName, attr.probability);
        }
      }
    }
    
    return dominantMap;
  }
  
  /**
   * Calcola la probabilità massima per una domanda basandosi sugli indici corrispondenti
   */
  private getQuestionProbability(questionId: number, attributes: Attribute[]): number {
    const indices = QUESTION_TO_INDICES[questionId];
    if (!indices || indices.length === 0) {
      return 0;
    }
    
    let maxProb = 0;
    for (const idx of indices) {
      const attr = attributes.find(a => a.index === idx);
      if (attr && attr.probability > maxProb) {
        maxProb = attr.probability;
      }
    }
    
    return maxProb;
  }
  
  /**
   * METODO CENTRALE: Classifica una singola immagine e restituisce array di risposte
   * Usa WebGPU se disponibile per accelerazione hardware
   */
  async classifyImageById(imageId: number, imageName?: string): Promise<QuestionAnswer[]> {
    if (!this.session) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }
    
    try {
      // Costruisci il path dell'immagine
      const paddedNumber = String(imageId).padStart(6, '0');
      const imageUrl = `${CONFIG.imageFolder}/${paddedNumber}.png`;
      
      // Pre-processa l'immagine
      const tensor = await this.preprocessImage(imageUrl);
      
      // Esegui l'inferenza (WebGPU o WASM a seconda della configurazione)
      const feeds: Record<string, ort.Tensor> = {};
      const inputName = this.session.inputNames && this.session.inputNames.length > 0 
        ? this.session.inputNames[0] 
        : 'input';
      
      if (inputName) {
        feeds[inputName] = tensor;
      }
      
      const output = await this.session.run(feeds);
      const outputName = this.session.outputNames && this.session.outputNames.length > 0 
        ? this.session.outputNames[0] 
        : Object.keys(output)[0];
      
      if (!outputName) {
        throw new Error('Unable to determine output name');
      }
      
      const outputTensor = output[outputName];
      if (!outputTensor) {
        throw new Error('Output tensor is undefined');
      }
      
      const predictions = outputTensor.data as ArrayLike<number>;
      
      // Crea gli attributi con sigmoide
      const attributes: Attribute[] = [];
      for (let i = 0; i < predictions.length && i < CONFIG.attributeNames.length; i++) {
        const pred = predictions[i];
        if (pred !== undefined && pred !== null) {
          const probability = 1 / (1 + Math.exp(-pred));
          attributes.push({
            name: CONFIG.attributeNames[i],
            probability,
            rawValue: pred,
            index: i
          });
        }
      }
      
      // Ottieni i tratti dominanti
      const dominantTraits = this.getDominantAttributes(attributes);
      
      // Costruisci l'array di risposte
      const answers: QuestionAnswer[] = [];
      const isVowel = this.startsWithVowel(imageName);
      
      for (let questionId = 0; questionId <= 18; questionId++) {
        if (questionId === 17) {
          // "Ha un nome che inizia con una vocale?" - usato per ethnicity
          answers.push({
            questionId,
            answer: isVowel,
            percentage: isVowel ? 100 : 0
          });
        } else if (questionId === 18) {
          // "Ha un nome che inizia con una consonante?"
          answers.push({
            questionId,
            answer: !isVowel,
            percentage: !isVowel ? 100 : 0
          });
        } else {
          // Attributi dal modello
          const indices = QUESTION_TO_INDICES[questionId];
          if (indices && indices.length > 0) {
            const probability = this.getQuestionProbability(questionId, attributes);
            const percentage = Math.round(probability * 100);
            
            // Determina se la risposta è true controllando i tratti dominanti
            let answer = false;
            for (const [displayName] of dominantTraits.entries()) {
              const displayIndices = indices.map(idx => 
                CONFIG.displayMapping[idx as keyof typeof CONFIG.displayMapping]
              );
              if (displayIndices.includes(displayName)) {
                answer = true;
                break;
              }
            }
            
            answers.push({
              questionId,
              answer,
              percentage
            });
          } else {
            // Attributo non presente nel nuovo modello
            answers.push({
              questionId,
              answer: false,
              percentage: 0
            });
          }
        }
      }
      
      return answers;
      
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      throw new Error(`Failed to classify image ${imageId}: ${message}`);
    }
  }
  
  /**
   * Restituisce il provider di esecuzione attualmente in uso
   */
  getExecutionProvider(): string {
    return this.executionProvider;
  }
  
  private startsWithVowel(name?: string): boolean {
    if (!name) return false;
    const firstChar = name.charAt(0).toLowerCase();
    return ['a', 'e', 'i', 'o', 'u'].includes(firstChar);
  }
}

/**
 * FUNZIONE DI UTILITÀ
 */
export async function classifyImageById(
  imageId: number,
  imageName?: string,
  options?: {
    modelPath?: string;
    useWebGpu?: boolean;
  }
): Promise<QuestionAnswer[]> {
  const classifier = new FacialAttributesClassifier();
  await classifier.loadModel(options?.useWebGpu ?? true, options?.modelPath);
  const answers = await classifier.classifyImageById(imageId, imageName);
  
  console.log(`Classification completed using: ${classifier.getExecutionProvider()}`);
  
  return answers;
}