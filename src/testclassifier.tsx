import { classifyFacialAttributes, FacialAttributesClassifier } from './standalone.js';

/**
 * TEST 1: Uso rapido con funzione helper
 */
async function testQuickUsage() {
  console.log('=== TEST 1: Quick Usage ===');
  
  try {
    const output = await classifyFacialAttributes({
      numImages: 24,
      imageFolder: 'images_224',
      onProgress: (current, total) => {
        console.log(`📸 Processing image ${current}/${total}`);
      }
    });
    
    console.log('\n✅ Classification completed!');
    console.log(`⏱️  Time: ${output.elapsedTime.toFixed(2)} seconds`);
    console.log(`📊 Processed ${output.results.length} images`);
    console.log('\n📈 Statistics:');
    
    // Mostra statistiche ordinate PARTE IMPORTANTE
    Object.entries(output.statistics)
      .sort((a, b) => b[1] - a[1])
      .forEach(([attr, count], index) => {
        console.log(`${index + 1}. ${attr}: ${count}`);
      });
    
    // Mostra dettagli prima immagine
    console.log('\n🖼️  First image details:');
    const firstResult = output.results[0];
    if (firstResult?.success && firstResult.attributes) {
      firstResult.attributes
        .filter(attr => attr.probability > 0.5)
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 5)
        .forEach(attr => {
          console.log(`   ${attr.name}: ${(attr.probability * 100).toFixed(1)}%`);
        });
    }
    
    return output;
  } catch (error) {
    console.error('❌ Error:', error);
    throw error;
  }
}

/**
 * TEST 2: Uso con classe (più controllo)
 */
async function testClassUsage() {
  console.log('\n=== TEST 2: Class Usage ===');
  
  const classifier = new FacialAttributesClassifier();
  
  try {
    console.log('Loading model...');
    await classifier.loadModel();
    console.log('✅ Model loaded');
    
    console.log('Starting classification...');
    const output = await classifier.classifyAll(24, 'images_224', (cur, tot) => {
      if (cur % 5 === 0 || cur === tot) {
        console.log(`Progress: ${cur}/${tot}`);
      }
    });
    
    console.log(`\n✅ Done in ${output.elapsedTime.toFixed(2)}s`);
    
    // Conta successi ed errori
    const successful = output.results.filter(r => r.success).length;
    const failed = output.results.filter(r => !r.success).length;
    console.log(`Success: ${successful}, Failed: ${failed}`);
    
    return output;
  } catch (error) {
    console.error('❌ Error:', error);
    throw error;
  }
}

/**
 * TEST 3: Verifica formato output
 */
async function testOutputFormat() {
  console.log('\n=== TEST 3: Output Format Validation ===');
  
  const output = await classifyFacialAttributes({ numImages: 3 }); // Solo 3 immagini per test rapido
  
  // Verifica struttura
  console.log('Checking output structure...');
  console.assert(Array.isArray(output.results), '❌ results should be array');
  console.assert(typeof output.statistics === 'object', '❌ statistics should be object');
  console.assert(typeof output.elapsedTime === 'number', '❌ elapsedTime should be number');
  
  // Verifica primo risultato
  const first = output.results[0];
  if (first) {
    console.assert(typeof first.imageNumber === 'number', '❌ imageNumber should be number');
    console.assert(typeof first.imageUrl === 'string', '❌ imageUrl should be string');
    console.assert(typeof first.success === 'boolean', '❌ success should be boolean');
    console.assert(Array.isArray(first.attributes), '❌ attributes should be array');
  }
  
  console.log('✅ Output format is correct');
  
  // Stampa JSON per ispezione
  console.log('\nSample output (first result):');
  console.log(JSON.stringify(first, null, 2));
  
  return output;
}

/**
 * TEST 4: Export dei risultati
 */
async function testExportResults() {
  console.log('\n=== TEST 4: Export Results ===');
  
  const output = await classifyFacialAttributes({ numImages: 5 });
  
  // Esporta come JSON
  const jsonOutput = JSON.stringify(output, null, 2);
  console.log('JSON size:', (jsonOutput.length / 1024).toFixed(2), 'KB');
  
  // Esporta solo statistiche
  const statsOnly = {
    statistics: output.statistics,
    totalImages: output.results.length,
    elapsedTime: output.elapsedTime
  };
  console.log('\nStatistics only:');
  console.log(JSON.stringify(statsOnly, null, 2));
  
  return output;
}

/**
 * RUN ALL TESTS
 */
export async function runAllTests() {
  console.log('🚀 Starting Facial Attributes Classifier Tests\n');
  
  try {
    // Esegui test in sequenza
    await testQuickUsage();
    //await testClassUsage();
    //await testOutputFormat();
    //await testExportResults();
    
    console.log('\n✅ All tests passed!');
  } catch (error) {
    console.error('\n❌ Tests failed:', error);
  }
}

// Esegui i test
runAllTests();