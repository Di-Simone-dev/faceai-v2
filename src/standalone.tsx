import * as ort from 'onnxruntime-web';

/**
 * CONFIGURAZIONE
 */
const CONFIG = {
  modelPath: 'facial_attributes_model.onnx',
  imageFolder: 'images_224',
  attributeNames: [
    'a man', 'a woman', 'a boy', 'a girl',
    'a person with blond hair', 'a person with brown hair', 'a person with black hair',
    'a person with red hair', 'a person with gray hair', 'a person with white hair',
    'a bald person', 'a bald person', 'a person with short hair',
    'a person with hair around his neck', 'a bald person', 'a person with straight hair',
    'a person with curly hair', 'a person with wavy hair', 'a person with afro hair',
    'clean shaven', 'stubble', 'beard',
    'a person with brown eyes', 'a person with blue eyes', 'a person with green eyes',
    'a person with hazel eyes', 'a person with black eyes', 'a person with amber eyes',
    'a person with clean hair', 'a person with few hair', 'a woman with her head covered',
    'a woman with visible forehead', 'a woman with a headband on her forehead',
    'a person with a hat', 'a person with eyes', 'a person with visible eyes',
    'a person with eye wrinkles', 'a person with eye bags', 'a person with eyeglasses'
  ],
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
  displayMapping: {
    0: 'Uomo', 1: 'Donna', 2: 'Uomo', 3: 'Donna',
    4: 'Capelli biondi', 5: 'Capelli Marroni', 6: 'Capelli neri',
    7: 'Capelli rossi', 8: 'Capelli grigi', 9: 'Capelli bianchi', 10: 'No Capelli',
    11: 'No capelli', 12: 'Capelli corti', 13: 'Capelli lunghi',
    14: 'No capelli', 15: 'Capelli lisci', 16: 'Capelli ricci',
    17: 'Capelli mossi', 18: 'Capelli afro',
    19: 'Senza Barba', 20: 'Con Barba', 21: 'Con Barba',
    22: 'Occhi marroni', 23: 'Occhi azzurri', 24: 'Occhi verdi',
    25: 'Occhi verdi', 26: 'Occhi marroni', 27: 'Occhi verdi',
    28: 'Senza Cappello', 29: 'Senza Cappello', 30: 'Con Cappello',
    31: 'Senza Cappello', 32: 'Con Cappello', 33: 'Con Cappello',
    34: 'Senza Occhiali', 35: 'Senza Occhiali', 36: 'Senza Occhiali',
    37: 'Senza Occhiali', 38: 'Con Occhiali'
  }
};

// Mapping domanda -> tratto -> indici attributi originali
const QUESTION_TO_INDICES: Record<number, number[]> = {
  0: [0, 2],           // Uomo
  1: [1, 3],           // Donna
  2: [4],              // Capelli biondi
  3: [5],              // Capelli Marroni
  4: [6],              // Capelli neri
  5: [7],              // Capelli rossi
  6: [10, 11, 14],     // No Capelli
  7: [13],             // Capelli lunghi
  8: [12],             // Capelli corti
  9: [16],             // Capelli ricci
  10: [15],            // Capelli lisci
  11: [20, 21],        // Con Barba
  12: [23],            // Occhi azzurri
  13: [24, 25, 27],    // Occhi verdi
  14: [22, 26],        // Occhi marroni
  15: [38],            // Con Occhiali
  16: [30, 32, 33]     // Con Cappello
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
 * CLASSE PRINCIPALE
 */
export class FacialAttributesClassifier {
  private session: ort.InferenceSession | null = null;
  
  async loadModel(modelPath?: string, dataPath?: string): Promise<void> {
    const mPath = modelPath || CONFIG.modelPath;
    const dPath = dataPath || `${CONFIG.modelPath}.data`;
    
    try {
      const modelResponse = await fetch(mPath);
      const modelBuffer = await modelResponse.arrayBuffer();
      
      const dataResponse = await fetch(dPath);
      const dataBuffer = await dataResponse.arrayBuffer();
      
      this.session = await ort.InferenceSession.create(modelBuffer, {
        externalData: [{ data: dataBuffer, path: dPath }]
      });
      
      console.log('Model loaded successfully');
    } catch (error) {
      throw new Error(`Failed to load model: ${error}`);
    }
  }
  
  private async preprocessImage(imageUrl: string): Promise<ort.Tensor> {
    return new Promise<ort.Tensor>((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas');
          canvas.width = 224;
          canvas.height = 224;
          const ctx = canvas.getContext('2d');
          
          if (!ctx) {
            reject(new Error('Unable to obtain 2D context'));
            return;
          }
          
          ctx.drawImage(img, 0, 0, 224, 224);
          const imageData = ctx.getImageData(0, 0, 224, 224);
          const data = imageData.data;
          
          const red: number[] = [], green: number[] = [], blue: number[] = [];
          
          for (let i = 0; i < data.length; i += 4) {
            red.push((data[i] ?? 0) / 255.0);
            green.push((data[i + 1] ?? 0) / 255.0);
            blue.push((data[i + 2] ?? 0) / 255.0);
          }
          
          const input = new Float32Array([...red, ...green, ...blue]);
          const tensor = new ort.Tensor('float32', input, [1, 3, 224, 224]);
          
          resolve(tensor);
        } catch (err) {
          reject(err);
        }
      };
      
      img.onerror = () => reject(new Error(`Failed to load image: ${imageUrl}`));
      img.src = imageUrl;
    });
  }
  
  private getDominantAttributes(attributes: Attribute[]): Map<string, number> {
    const dominantMap = new Map<string, number>();
    
    // Processa i gruppi di attributi
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
   * METODO CENTRALE: Classifica una singola immagine e restituisce array di 19 risposte
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
      
      // Esegui l'inferenza
      const feeds: Record<string, ort.Tensor> = {};
      const inputName = this.session.inputNames[0];
      if (inputName) {
        feeds[inputName] = tensor;
      }
      
      const output = await this.session.run(feeds);
      const outputName = this.session.outputNames[0];
      
      if (!outputName) {
        throw new Error('Unable to determine output name');
      }
      
      const outputTensor = output[outputName];
      if (!outputTensor) {
        throw new Error('Output tensor is undefined');
      }
      
      const predictions = outputTensor.data as ArrayLike<number>;
      
      // Crea gli attributi
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
      
      // Ottieni i tratti dominanti (per determinare answer: true/false)
      const dominantTraits = this.getDominantAttributes(attributes);
      
      // Costruisci l'array di 19 risposte (0-18)
      const answers: QuestionAnswer[] = [];
      const isVowel = this.startsWithVowel(imageName);
      
      for (let questionId = 0; questionId <= 18; questionId++) {
        if (questionId === 17) {
          // "Ha un nome che inizia con una vocale?"
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
          // Attributi canonici
          // answer: true se è dominante
          // percentage: sempre la probabilità reale (anche se non dominante)
          const indices = QUESTION_TO_INDICES[questionId];
          if (indices) {
            const probability = this.getQuestionProbability(questionId, attributes);
            const percentage = Math.round(probability * 100);
            
            // Determina se la risposta è true controllando se corrisponde a un tratto dominante
            const trait = Object.keys(QUESTION_TO_INDICES).find(key => 
              parseInt(key) === questionId
            );
            let answer = false;
            if (trait) {
              // Cerca il display name corrispondente verificando gli indici
              for (const [displayName, prob] of dominantTraits.entries()) {
                // Verifica se questo displayName corrisponde a questa domanda
                const displayIndices = indices.map(idx => 
                  CONFIG.displayMapping[idx as keyof typeof CONFIG.displayMapping]
                );
                if (displayIndices.includes(displayName)) {
                  answer = true;
                  break;
                }
              }
            }
            
            answers.push({
              questionId,
              answer,
              percentage
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
    dataPath?: string;
  }
): Promise<QuestionAnswer[]> {
  const classifier = new FacialAttributesClassifier();
  await classifier.loadModel(options?.modelPath, options?.dataPath);
  return await classifier.classifyImageById(imageId, imageName);
}