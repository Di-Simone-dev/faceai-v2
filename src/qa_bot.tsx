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
 * MAPPING DOMANDE -> TRATTI
 */
const QUESTION_TO_TRAIT_MAPPING: Record<number, string> = {
  0: 'Uomo',                     // "È un uomo?"
  1: 'Donna',                    // "È una donna?"
  2: 'Capelli biondi',           // "Ha i capelli biondi?"
  3: 'Capelli Marroni',          // "Ha i capelli castani?"
  4: 'Capelli neri',             // "Ha i capelli neri?"
  5: 'Capelli rossi',            // "Ha i capelli rossi?"
  6: 'No Capelli',               // "È calvo/a?"
  7: 'Capelli lunghi',           // "Ha i capelli lunghi?"
  8: 'Capelli corti',            // "Ha i capelli corti?"
  9: 'Capelli ricci',            // "Ha i capelli ricci?"
  10: 'Capelli lisci',           // "Ha i capelli lisci?"
  11: 'Con Barba',               // "Ha la barba?"
  12: 'Occhi azzurri',           // "Ha gli occhi azzurri?"
  13: 'Occhi verdi',             // "Ha gli occhi verdi?"
  14: 'Occhi marroni',           // "Ha gli occhi marroni?"
  15: 'Con Occhiali',            // "Porta gli occhiali?"
  16: 'Con Cappello',            // "Porta un cappello?"
  17: 'Vocale',                  // "Ha un nome che inizia con una vocale?"
  18: 'Consonante'               // "Ha un nome che inizia con una consonante?"
};

/**
 * TIPI
 */
export type DominantAttribute = {
  group: string;
  attribute: string;
  probability: number;
  percentage: number;
  index: number;
};

export type ImageDominantAttributes = {
  imageId: number;
  imageUrl: string;
  dominantAttributes: DominantAttribute[];
};

export type DominantAttributesJSON = {
  totalImages: number;
  elapsedTime: number;
  images: ImageDominantAttributes[];
};

export type TraitCheckResult = {
  hasTrait: boolean;
  percentage: number;
};

/**
 * FUNZIONI INTERNE
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

async function generateDominantAttributesJSON(
  numImages: number = 24,
  imageFolder: string = 'images_224'
): Promise<DominantAttributesJSON> {
  const output = await classifyFacialAttributes({
    numImages,
    imageFolder,
    onProgress: (current: number, total: number) => {
      console.log(`📸 Processing ${current}/${total}`);
    }
  });
  
  const images = output.results.map(result => extractDominantAttributes(result));
  
  return {
    totalImages: output.results.length,
    elapsedTime: output.elapsedTime,
    images
  };
}

function hasTrait(
  json: DominantAttributesJSON,
  imageId: number,
  trait: string
): TraitCheckResult {
  const image = json.images.find(img => img.imageId === imageId);
  
  if (!image) {
    console.warn(`Image ${imageId} not found`);
    return { hasTrait: false, percentage: 0 };
  }
  
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
 * FUNZIONE PRINCIPALE ESPORTATA
 * 
 * Controlla se un'immagine ha un tratto specifico usando l'indice della domanda
 * 
 * @param imageId - ID dell'immagine (1-24)
 * @param questionIndex - Indice della domanda (0-18)
 * @param characterName - Nome del personaggio (necessario solo per domande 17-18)
 * @param json - JSON precaricato (opzionale, se non fornito verrà generato)
 * @returns { hasTrait: boolean, percentage: number } o null se errore
 * 
 * @example
 * // Con JSON precaricato (consigliato per performance)
 * const json = await generateJSON();
 * const result = await hasTraitByQuestion(1, 0, 'Mario', json);  // { hasTrait: true, percentage: 92 }
 * 
 * // Senza JSON precaricato (genera al volo)
 * const result = await hasTraitByQuestion(1, 0, 'Mario');
 */
export async function hasTraitByQuestion(
  imageId: number,
  questionIndex: number,
  characterName: string,
  json?: DominantAttributesJSON
): Promise<TraitCheckResult | null> {
  // Genera il JSON se non è stato fornito
  if (!json) {
    console.log('⚠️ JSON not provided, generating now (this may take time)...');
    json = await generateDominantAttributesJSON();
  }
  
  // Ottieni il tratto dal mapping
  const trait = QUESTION_TO_TRAIT_MAPPING[questionIndex];
  
  if (!trait) {
    console.warn(`Question ${questionIndex} is not mapped`);
    return null;
  }
  
  // Gestisci domande sui tratti visivi (0-16)
  if (questionIndex >= 0 && questionIndex <= 16) {
    return hasTrait(json, imageId, trait);
  }
  
  // Gestisci domande sul nome (17-18)
  const firstName = characterName.trim()[0]?.toLowerCase();
  
  if (!firstName) {
    console.warn('Invalid character name');
    return { hasTrait: false, percentage: 0 };
  }
  
  const isVowel = ['a', 'e', 'i', 'o', 'u'].includes(firstName);
  
  if (questionIndex === 17) {
    // "Ha un nome che inizia con una vocale?"
    return {
      hasTrait: isVowel,
      percentage: 100
    };
  }
  
  if (questionIndex === 18) {
    // "Ha un nome che inizia con una consonante?"
    return {
      hasTrait: !isVowel,
      percentage: 100
    };
  }
  
  return null;
}

/**
 * FUNZIONE HELPER: Genera e ritorna il JSON
 * Utile per precaricare il JSON e riutilizzarlo con hasTraitByQuestion
 */
export async function generateJSON(
  numImages: number = 24,
  imageFolder: string = 'images_224'
): Promise<DominantAttributesJSON> {
  console.log('🚀 Generating dominant attributes JSON...');
  return await generateDominantAttributesJSON(numImages, imageFolder);
}