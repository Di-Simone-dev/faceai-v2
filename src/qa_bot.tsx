import { classifyFacialAttributes, type ClassificationResult, type Attribute } from './standalone.js';

/**
 * CONFIGURAZIONE
 */
const CONFIG = {
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

/**
 * TIPO: Attributo dominante con tutte le info
 */
export type DominantAttribute = {
  group: string;
  attribute: string;
  probability: number;
  percentage: number;
  index: number;
};

/**
 * TIPO: Attributi dominanti per una singola immagine
 */
export type ImageDominantAttributes = {
  imageId: number;
  imageUrl: string;
  dominantAttributes: DominantAttribute[];
};

/**
 * TIPO: Output JSON completo
 */
export type DominantAttributesJSON = {
  totalImages: number;
  elapsedTime: number;
  images: ImageDominantAttributes[];
};

/**
 * Trova l'attributo dominante in un gruppo
 */
function getDominantInGroup(
  attributes: Attribute[], 
  groupName: string,
  groupIndices: number[]
): DominantAttribute | null {
  let maxProb = -1;
  let dominant: Attribute | null = null;
  
  for (const idx of groupIndices) {
    const attr = attributes.find(a => a.index === idx);
    if (attr && attr.probability > maxProb) {
      maxProb = attr.probability;
      dominant = attr;
    }
  }
  
  if (!dominant) {
    return null;
  }
  
  const displayName = CONFIG.displayMapping[dominant.index as keyof typeof CONFIG.displayMapping] || dominant.name || 'Unknown';
  
  return {
    group: groupName,
    attribute: displayName,
    probability: dominant.probability,
    percentage: Math.round(dominant.probability * 100),
    index: dominant.index
  };
}

/**
 * Estrae tutti gli attributi dominanti da un risultato di classificazione
 */
function extractDominantAttributes(result: ClassificationResult): ImageDominantAttributes {
  const dominantAttributes: DominantAttribute[] = [];
  
  if (result.success && result.attributes) {
    CONFIG.attributeGroups.forEach(group => {
      const dominant = getDominantInGroup(result.attributes!, group.name, group.indices);
      if (dominant) {
        dominantAttributes.push(dominant);
      }
    });
  }
  
  return {
    imageId: result.imageNumber,
    imageUrl: result.imageUrl,
    dominantAttributes
  };
}

/**
 * FUNZIONE PRINCIPALE: Genera JSON con tutti gli attributi dominanti
 * 
 * @param numImages - Numero di immagini da processare (default: 24)
 * @param imageFolder - Cartella delle immagini (default: 'images_224')
 * @returns JSON con tutti gli attributi dominanti per ogni immagine
 */
export async function generateDominantAttributesJSON(
  numImages: number = 24,
  imageFolder: string = 'images_224'
): Promise<DominantAttributesJSON> {
  console.log('🚀 Starting classification...');
  
  // Classifica tutte le immagini
  const output = await classifyFacialAttributes({
    numImages,
    imageFolder,
    onProgress: (current: number, total:number) => {
      console.log(`📸 Processing ${current}/${total}`);
    }
  });
  
  console.log('✅ Classification completed!');
  console.log('📊 Extracting dominant attributes...');
  
  // Estrai attributi dominanti per ogni immagine
  const images = output.results.map(result => extractDominantAttributes(result));
  
  const json: DominantAttributesJSON = {
    totalImages: output.results.length,
    elapsedTime: output.elapsedTime,
    images
  };
  
  console.log('✅ JSON generated!');
  
  return json;
}

/**
 * FUNZIONE UTILITÀ: Salva il JSON come stringa formattata
 */
export function jsonToString(json: DominantAttributesJSON, prettify: boolean = true): string {
  return JSON.stringify(json, null, prettify ? 2 : 0);
}

/**
 * FUNZIONE UTILITÀ: Download del JSON come file
 */
export function downloadJSON(json: DominantAttributesJSON, filename: string = 'dominant_attributes.json'): void {
  const jsonString = jsonToString(json, true);
  const blob = new Blob([jsonString], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  URL.revokeObjectURL(url);
  console.log(`✅ JSON downloaded as ${filename}`);
}

/**
 * FUNZIONE UTILITÀ: Trova un'immagine specifica nel JSON
 */
export function findImageById(json: DominantAttributesJSON, imageId: number): ImageDominantAttributes | undefined {
  return json.images.find(img => img.imageId === imageId);
}

/**
 * FUNZIONE UTILITÀ: Filtra immagini per attributo specifico
 */
export function filterByAttribute(
  json: DominantAttributesJSON, 
  groupName: string, 
  attributeValue: string
): ImageDominantAttributes[] {
  return json.images.filter(img => {
    const attr = img.dominantAttributes.find(a => a.group === groupName);
    return attr && attr.attribute === attributeValue;
  });
}

/**
 * MAPPING DOMANDE -> TRATTI
 * Ogni indice corrisponde a una domanda e al relativo tratto da verificare
 */
const QUESTION_TO_TRAIT_MAPPING: Record<number, string | null> = {
  0: 'Uomo',                    // "È un uomo?"
  1: 'Donna',                   // "È una donna?"
  2: 'Capelli biondi',           // "Ha i capelli biondi?"
  3: 'Capelli marroni',          // "Ha i capelli castani?"
  4: 'Capelli neri',             // "Ha i capelli neri?"
  5: 'Capelli rossi',            // "Ha i capelli rossi?"
  6: 'No Capelli',               // "È calvo/a?"
  7: 'Capelli lunghi',           // "Ha i capelli lunghi?"
  8: 'Capelli corti',            // "Ha i capelli corti?"
  9: 'Capelli ricci',            // "Ha i capelli ricci?"
  10: 'Capelli lisci',            // "Ha i capelli lisci?"
  11: 'Con Barba',                // "Ha la barba?"
  12: 'Occhi azzurri',           // "Ha gli occhi azzurri?"
  13: 'Occhi verdi',             // "Ha gli occhi verdi?"
  14: 'Occhi marroni',           // "Ha gli occhi marroni?"
  15: 'Con Occhiali',            // "Porta gli occhiali?"
  16: 'Con Cappello',            // "Porta un cappello?"
  17: 'Vocale',                      // "Ha un nome che inizia con una vocale?" - non classificabile
  18: 'Consonante'                       // "Ha un nome che inizia con una consonante?" - non classificabile
};

/**
 * TIPO: Risultato del controllo tratto
 */
export type TraitCheckResult = {
  hasTrait: boolean;
  percentage: number;
};

/**
 * FUNZIONE PRINCIPALE: Controlla se un'immagine ha un tratto dominante specifico
 * 
 * @param json - JSON con tutti gli attributi dominanti
 * @param imageId - ID dell'immagine (1-24)
 * @param trait - Tratto da verificare (es: 'Uomo', 'Capelli neri', 'Con Occhiali')
 * @returns Oggetto con hasTrait (boolean) e percentage (number)
 * 
 * @example
 * const json = await generateDominantAttributesJSON();
 * hasTrait(json, 1, 'Uomo');           // { hasTrait: true, percentage: 92 }
 * hasTrait(json, 5, 'Con Occhiali');   // { hasTrait: false, percentage: 0 }
 * hasTrait(json, 12, 'Capelli neri');  // { hasTrait: true, percentage: 87 }
 */
export function hasTrait(
  json: DominantAttributesJSON,
  imageId: number,
  trait: string
): TraitCheckResult {
  // Trova l'immagine
  const image = json.images.find(img => img.imageId === imageId);
  
  if (!image) {
    console.warn(`Image ${imageId} not found`);
    return { hasTrait: false, percentage: 0 };
  }
  
  // Cerca il tratto negli attributi dominanti
  const attribute = image.dominantAttributes.find(attr => attr.attribute === trait);
  
  if (attribute) {
    return {
      hasTrait: true,
      percentage: attribute.percentage
    };
  }
  
  return { hasTrait: false, percentage: 0 };
}

/**
 * FUNZIONE CON INDICE DOMANDA: Controlla un tratto usando l'indice della domanda
 * 
 * @param json - JSON con tutti gli attributi dominanti
 * @param imageId - ID dell'immagine (1-24)
 * @param questionIndex - Indice della domanda (0-19)
 * @returns Oggetto con hasTrait (boolean) e percentage (number), o null se la domanda non è classificabile
 * 
 * @example
 * const json = await generateDominantAttributesJSON();
 * hasTraitByQuestion(json, 1, 0);   // { hasTrait: true, percentage: 85 } - capelli biondi
 * hasTraitByQuestion(json, 1, 16);  // { hasTrait: true, percentage: 92 } - uomo
 * hasTraitByQuestion(json, 1, 14);  // { hasTrait: false, percentage: 0 } - occhiali
 * hasTraitByQuestion(json, 1, 18);  // null - domanda non classificabile
 */
export function hasTraitByQuestion(
  json: DominantAttributesJSON,
  imageId: number,
  questionIndex: number
): TraitCheckResult | null {
  // Ottieni il tratto dal mapping
  const trait = QUESTION_TO_TRAIT_MAPPING[questionIndex];
  const name:string = "Valerio"  //TEMPORANEO, DA SOSTITUIRE
  // Se la domanda non è mappabile a un tratto visivo, ritorna null 
  if (trait === null || trait === undefined) {
    console.warn(`Question ${questionIndex} is not classifiable`);
    return null;
  }
  if(questionIndex>-1 && questionIndex <17){  //nel caso di tratti visivi
    return hasTrait(json, imageId, trait);
  }
  else return {
      hasTrait: /^[aeiou]/i.test(name),  //metodo veloce case insensitive regex 
      percentage: 100
    }; 
  // Usa la funzione hasTrait con il tratto mappato
  
}

/**
 * FUNZIONE BATCH: Verifica multiple domande per un'immagine
 * 
 * @param json - JSON con tutti gli attributi dominanti
 * @param imageId - ID dell'immagine (1-24)
 * @param questionIndices - Array di indici di domande da verificare
 * @returns Array di risultati (null per domande non classificabili)
 * 
 * @example
 * hasTraitsByQuestions(json, 1, [0, 2, 14, 16]);
 * // [
 * //   { hasTrait: false, percentage: 15 },  // capelli biondi
 * //   { hasTrait: true, percentage: 87 },   // capelli neri
 * //   { hasTrait: true, percentage: 91 },   // occhiali
 * //   { hasTrait: true, percentage: 92 }    // uomo
 * // ]
 */
export function hasTraitsByQuestions(
  json: DominantAttributesJSON,
  imageId: number,
  questionIndices: number[]
): (TraitCheckResult | null)[] {
  return questionIndices.map(index => hasTraitByQuestion(json, imageId, index));
}

/**
 * FUNZIONE UTILITÀ: Ottieni il tratto associato a un indice domanda
 * 
 * @param questionIndex - Indice della domanda (0-19)
 * @returns Il tratto associato o null se non classificabile
 */
export function getTraitForQuestion(questionIndex: number): string | null {
  return QUESTION_TO_TRAIT_MAPPING[questionIndex] ?? null;
}

/**
 * ESEMPIO DI USO
 */
export async function example() {
  // Genera il JSON
  const json = await generateDominantAttributesJSON(24, 'images_224');
  
  // Stampa il JSON
  console.log('📄 JSON Output:');
  console.log(jsonToString(json));
  
  // Download del file
  downloadJSON(json);
  
  // ===== ESEMPIO FUNZIONE hasTrait =====
  console.log('\n🔍 Testing hasTrait function:');
  
  const result1 = hasTrait(json, 1, 'Uomo');
  console.log('Image 1 is Uomo?', result1);
  
  // ===== ESEMPIO FUNZIONE hasTraitByQuestion =====
  console.log('\n🔍 Testing hasTraitByQuestion (con indici):');
  
  const q0 = hasTraitByQuestion(json, 1, 0);   // "È un uomo?"
  console.log('Q0 - È un uomo?', q0);

  const q1 = hasTraitByQuestion(json, 1, 1);   // "È un uomo?"
  console.log('Q1 - È una donna?', q0);
  
  const q2 = hasTraitByQuestion(json, 1, 2);   // "Ha i capelli neri?"
  console.log('Q2 - Ha i capelli biondi?', q2);
  
  const q17 = hasTraitByQuestion(json, 1, 17); // "Ha un nome che inizia con una vocale?"
  console.log('Q17 - Vocale?', q17); 

  const q18 = hasTraitByQuestion(json, 1, 18); // "Ha un nome che inizia con una vocale?"
  console.log('Q18 - Consonante?', q18); 
  
  // ===== ESEMPIO BATCH =====
 /* console.log('\n🔍 Testing batch questions:');
  const batchResults = hasTraitsByQuestions(json, 1, [0, 2, 14, 16]);
  console.log('Batch results:', batchResults);
  
  // Query avanzate
  const image5 = findImageById(json, 5);
  console.log('\nImage 5 attributes:', image5);
  
  const menImages = filterByAttribute(json, 'Gender', 'Uomo');
  console.log(`\nFound ${menImages.length} men`);
  
  const withGlasses = filterByAttribute(json, 'Eyeglasses', 'Con Occhiali');
  console.log(`Found ${withGlasses.length} people with glasses`); */
  
  return json;
}