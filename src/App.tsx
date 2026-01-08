import { useState } from 'react';
import { hasTraitByQuestion, generateJSON } from './qa_bot.js';

type DominantAttributesJSON = any; // Replace 'any' with the actual type definition

function App() {
  const [jsonData, setJsonData] = useState<DominantAttributesJSON | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const runTests = async () => {
    setIsLoading(true);
    
    try {
      // Genero il JSON una volta all'inizio
      console.log('🚀 Generating JSON...');
      const json = await generateJSON(24, 'images_224');
      console.log('✅ JSON generated!', json);
      setJsonData(json);

      // Riuso il JSON
      const result1 = await hasTraitByQuestion(1, 0, 'Mario', json);
      console.log('Result 1 (Image 1, Q0 - È un uomo?):', result1);

      const result2 = await hasTraitByQuestion(1, 15, 'Mario', json);
      console.log('Result 2 (Image 1, Q15 - Porta occhiali?):', result2);

      const result3 = await hasTraitByQuestion(5, 4, 'Luigi', json);
      console.log('Result 3 (Image 5, Q4 - Capelli neri?):', result3);

      const result4 = await hasTraitByQuestion(12, 17, 'Anna', json);
      console.log('Result 4 (Image 12, Q17 - Nome con vocale?):', result4);

      // Puoi anche fare batch
      console.log('\n🔄 Testing batch...');
      const allResults = await Promise.all([
        hasTraitByQuestion(1, 0, 'Mario', json),
        hasTraitByQuestion(1, 1, 'Mario', json),
        hasTraitByQuestion(1, 2, 'Mario', json)
      ]);
      console.log('Batch results:', allResults);
      
      console.log('✅ All tests completed!');
    } catch (error) {
      console.error('❌ Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">QA Bot Tester</h1>
      <button 
        onClick={runTests}
        disabled={isLoading}
        className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
      >
        {isLoading ? 'Running tests...' : 'Run Tests'}
      </button>
      <p className="mt-4 text-gray-600">
        {isLoading ? 'Check console for progress...' : 'Click the button to start tests'}
      </p>
    </div>
  );
}

export default App;