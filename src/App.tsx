
import { useEffect } from 'react';
import { runAllTests } from './testclassifier.js';

function App() {
  useEffect(() => {
    runAllTests();
  }, []); // Esegue solo al mount

  return (
    <div className="p-8">
      <h1>Testing in console...</h1>
    </div>
  );
}

export default App;