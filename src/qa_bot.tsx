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
 * FUNZIONE PRINCIPALE: Controlla se un'immagine ha un tratto dominante specifico
 * 
 * @param json - JSON con tutti gli attributi dominanti
 * @param imageId - ID dell'immagine (1-24)
 * @param trait - Tratto da verificare (es: 'Uomo', 'Capelli neri', 'Con Occhiali')
 * @returns true se il tratto è dominante, false altrimenti
 * 
 * @example
 * const json = await generateDominantAttributesJSON();
 * hasTrait(json, 1, 'Uomo');           // true o false
 * hasTrait(json, 5, 'Con Occhiali');   // true o false
 * hasTrait(json, 12, 'Capelli neri');  // true o false
 */
export function hasTrait(
  json: DominantAttributesJSON,
  imageId: number,
  trait: string
): boolean {
  // Trova l'immagine
  const image = json.images.find(img => img.imageId === imageId);
  
  if (!image) {
    console.warn(`Image ${imageId} not found`);
    return false;
  }
  
  // Cerca il tratto negli attributi dominanti
  return image.dominantAttributes.some(attr => attr.attribute === trait);
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
  
  console.log('Image 1 is Uomo?', hasTrait(json, 1, 'Uomo'));
  /*console.log('Image 1 has Occhiali?', hasTrait(json, 1, 'Con Occhiali'));
  console.log('Image 5 has Cappello?', hasTrait(json, 5, 'Con Cappello'));
  console.log('Image 12 has Capelli neri?', hasTrait(json, 12, 'Capelli neri'));
  
  // Query avanzate
  const image5 = findImageById(json, 5);
  console.log('\nImage 5 attributes:', image5);
  
  const menImages = filterByAttribute(json, 'Gender', 'Uomo');
  console.log(`\nFound ${menImages.length} men`);
  
  const withGlasses = filterByAttribute(json, 'Eyeglasses', 'Con Occhiali');
  console.log(`Found ${withGlasses.length} people with glasses`);*/
  
  return json;
}