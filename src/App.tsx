
import { useEffect } from 'react';
import { runAllTests } from './testclassifier.js';
import { example } from './qa_bot.js';  

function App() {
  useEffect(() => {
    runAllTests();
    example();
  }, []); // Esegue solo al mount

  return (
    <div className="p-8">
      <h1>Testing in console...</h1>
    </div>
  );
}

export default App;