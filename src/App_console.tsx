import { useEffect, useState } from 'react';
import { classifyImageById } from './standalone.js';
import type { QuestionAnswer } from './standalone.js';

export function App() {
  const [risultati, setRisultati] = useState<QuestionAnswer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchClassification() {
      try {
        setLoading(true);
        // Classifica l'immagine con ID 1, nome "Alice"
        const data = await classifyImageById(1, "Alice");
        
        console.log("Risultati classificazione:");
        data.forEach(({ questionId, answer, percentage }) => {
          console.log(`Domanda ${questionId}: ${answer ? 'SÌ' : 'NO'} (${percentage}%)`);
        });
        
        setRisultati(data);
      } catch (err) {
        console.error("Errore durante la classificazione:", err);
        setError(err instanceof Error ? err.message : 'Errore sconosciuto');
      } finally {
        setLoading(false);
      }
    }

    fetchClassification();
  }, []); // Array vuoto = esegui solo al mount

  return (
    <div className="p-8">
      {loading && <p>Caricamento classificazione...</p>}
      {error && <p className="text-red-500">Errore: {error}</p>}
      {!loading && !error && (
        <div>
          <h2 className="text-xl font-bold mb-4">Risultati Classificazione</h2>
          <ul className="space-y-2">
            {risultati.map(({ questionId, answer, percentage }) => (
              <li key={questionId}>
                Domanda {questionId}: {answer ? 'SÌ' : 'NO'} ({percentage}%)
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;