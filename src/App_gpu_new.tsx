import React, { useState } from 'react';
import { 
  FacialAttributesClassifier, 
  classifyImage, 
  classifyImageById,
  type QuestionAnswer 
} from './OnnxImageClassifierStandalone.js';

/**
 * APP.TSX - Componente React di esempio per il classificatore standalone
 * 
 * Questo componente mostra come integrare FacialAttributesClassifier
 * in un'applicazione React con vari esempi di utilizzo.
 */

interface ClassificationResult {
  imageName: string;
  answers: QuestionAnswer[];
  timestamp: number;
}

function App() {
  const [classifier, setClassifier] = useState<FacialAttributesClassifier | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [results, setResults] = useState<ClassificationResult[]>([]);
  const [currentExample, setCurrentExample] = useState<string>('');
  const [executionProvider, setExecutionProvider] = useState<string>('');

  /**
   * Inizializza il classificatore e carica il modello
   */
  const initializeClassifier = async (useWebGpu: boolean = true) => {
    setIsLoading(true);
    setCurrentExample('Inizializzazione modello...');
    
    try {
      const newClassifier = new FacialAttributesClassifier();
      await newClassifier.loadModel(useWebGpu, 'model_webgpu.onnx');
      
      setClassifier(newClassifier);
      setModelLoaded(true);
      setExecutionProvider(newClassifier.getExecutionProvider());
      setCurrentExample(`✅ Modello caricato con successo (${newClassifier.getExecutionProvider()})`);
      
      console.log('✅ Classificatore inizializzato');
    } catch (error) {
      console.error('❌ Errore inizializzazione:', error);
      setCurrentExample(`❌ Errore: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * ESEMPIO 1: Classificazione singola immagine
   */
  const example1_SingleImage = async () => {
    if (!classifier) {
      alert('Carica prima il modello!');
      return;
    }

    setIsLoading(true);
    setCurrentExample('ESEMPIO 1: Classificazione singola immagine');
    
    try {
      console.log('\n🎯 ESEMPIO 1: Classificazione singola immagine\n');
      
      const answers = await classifier.classifyImage(
        'images_256/000001.png',
        'Anna'
      );
      
      setResults([{
        imageName: 'Anna',
        answers,
        timestamp: Date.now()
      }]);
      
      setCurrentExample('✅ ESEMPIO 1 completato');
      
    } catch (error) {
      console.error('❌ Errore:', error);
      setCurrentExample(`❌ Errore ESEMPIO 1: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * ESEMPIO 2: Classificazione per ID
   */
  const example2_ClassifyById = async () => {
    if (!classifier) {
      alert('Carica prima il modello!');
      return;
    }

    setIsLoading(true);
    setCurrentExample('ESEMPIO 2: Classificazione per ID immagine');
    
    try {
      console.log('\n🎯 ESEMPIO 2: Classificazione per ID\n');
      
      const answers = await classifier.classifyImageById(1, 'Mario');
      
      setResults([{
        imageName: 'Mario (ID: 1)',
        answers,
        timestamp: Date.now()
      }]);
      
      setCurrentExample('✅ ESEMPIO 2 completato');
      
    } catch (error) {
      console.error('❌ Errore:', error);
      setCurrentExample(`❌ Errore ESEMPIO 2: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * ESEMPIO 3: Batch processing
   */
  const example3_BatchProcessing = async () => {
    if (!classifier) {
      alert('Carica prima il modello!');
      return;
    }

    setIsLoading(true);
    setCurrentExample('ESEMPIO 3: Batch processing (5 immagini)');
    
    try {
      console.log('\n🎯 ESEMPIO 3: Batch processing\n');
      
      const images = [
        { id: 1, name: 'Anna' },
        { id: 2, name: 'Bruno' },
        { id: 3, name: 'Elena' },
        { id: 4, name: 'Omar' },
        { id: 5, name: 'Irene' }
      ];
      
      const batchResults: ClassificationResult[] = [];
      
      for (const img of images) {
        setCurrentExample(`Processando ${img.name} (${img.id}/5)...`);
        const answers = await classifier.classifyImageById(img.id, img.name);
        batchResults.push({
          imageName: img.name,
          answers,
          timestamp: Date.now()
        });
      }
      
      setResults(batchResults);
      setCurrentExample(`✅ ESEMPIO 3 completato (${batchResults.length} immagini)`);
      
      // Statistiche
      const withBeard = batchResults.filter(r => {
        const beardQ = r.answers.find(qa => qa.questionId === 16);
        return beardQ?.answer === true;
      });
      
      console.log(`📊 Immagini con barba: ${withBeard.length}/${images.length}`);
      
    } catch (error) {
      console.error('❌ Errore:', error);
      setCurrentExample(`❌ Errore ESEMPIO 3: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * ESEMPIO 4: Filtraggio alta confidenza
   */
  const example4_HighConfidence = async () => {
    if (!classifier) {
      alert('Carica prima il modello!');
      return;
    }

    setIsLoading(true);
    setCurrentExample('ESEMPIO 4: Filtraggio alta confidenza (>70%)');
    
    try {
      console.log('\n🎯 ESEMPIO 4: Filtraggio alta confidenza\n');
      
      const answers = await classifier.classifyImage('images_256/000001.png', 'Test');
      
      // Filtra solo risposte con alta confidenza
      const highConfidence = answers.filter(qa => qa.percentage > 70);
      
      console.log(`\n🎯 ${highConfidence.length} attributi con confidenza > 70%:`);
      highConfidence.forEach(qa => {
        const answerStr = qa.answer ? '✓ Sì' : '✗ No';
        console.log(`   Q${qa.questionId}: ${answerStr} → ${qa.percentage}%`);
      });
      
      setResults([{
        imageName: 'Test (Alta confidenza)',
        answers: highConfidence,
        timestamp: Date.now()
      }]);
      
      setCurrentExample(`✅ ESEMPIO 4 completato (${highConfidence.length} attributi)`);
      
    } catch (error) {
      console.error('❌ Errore:', error);
      setCurrentExample(`❌ Errore ESEMPIO 4: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * ESEMPIO 5: Analisi dettagliata con raggruppamento
   */
  const example5_DetailedAnalysis = async () => {
    if (!classifier) {
      alert('Carica prima il modello!');
      return;
    }

    setIsLoading(true);
    setCurrentExample('ESEMPIO 5: Analisi dettagliata con raggruppamento');
    
    try {
      console.log('\n🎯 ESEMPIO 5: Analisi dettagliata\n');
      
      const answers = await classifier.classifyImage('images_256/000001.png', 'Alice');
      
      // Mappa delle domande
      const questionMap: Record<number, string> = {
        0: 'Sorriso', 1: 'Uomo', 2: 'Donna',
        3: 'Capelli Marroni', 4: 'Capelli Neri', 5: 'Capelli Biondi', 6: 'Capelli Grigi',
        7: 'Capelli Lunghi', 8: 'Capelli Corti',
        9: 'Asiatico', 10: 'Nero', 11: 'Latino', 12: 'Bianco',
        13: 'Occhi Azzurri', 14: 'Occhi Marroni', 15: 'Occhi Verdi',
        16: 'Con Barba', 17: 'Con Occhiali',
        18: 'Nome Vocale', 19: 'Nome Consonante'
      };
      
      // Raggruppa per categoria
      const groups = {
        'Genere': [1, 2],
        'Capelli - Colore': [3, 4, 5, 6],
        'Capelli - Lunghezza': [7, 8],
        'Etnia': [9, 10, 11, 12],
        'Occhi': [13, 14, 15],
        'Altro': [0, 16, 17],
        'Nome': [18, 19]
      };
      
      console.log('\n📋 Profilo completo:');
      console.log('═'.repeat(60));
      
      for (const [groupName, questionIds] of Object.entries(groups)) {
        console.log(`\n${groupName}:`);
        questionIds.forEach(qId => {
          const qa = answers.find(r => r.questionId === qId);
          if (qa) {
            const label = questionMap[qId] || `Q${qId}`;
            const status = qa.answer ? '✓' : '✗';
            console.log(`   ${status} ${label.padEnd(20)} ${qa.percentage}%`);
          }
        });
      }
      
      setResults([{
        imageName: 'Alice (Analisi dettagliata)',
        answers,
        timestamp: Date.now()
      }]);
      
      setCurrentExample('✅ ESEMPIO 5 completato');
      
    } catch (error) {
      console.error('❌ Errore:', error);
      setCurrentExample(`❌ Errore ESEMPIO 5: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * ESEMPIO 6: Comparazione tra due immagini
   */
  const example6_CompareImages = async () => {
    if (!classifier) {
      alert('Carica prima il modello!');
      return;
    }

    setIsLoading(true);
    setCurrentExample('ESEMPIO 6: Comparazione tra immagini');
    
    try {
      console.log('\n🎯 ESEMPIO 6: Comparazione immagini\n');
      
      const img1 = await classifier.classifyImageById(1, 'Persona A');
      const img2 = await classifier.classifyImageById(2, 'Persona B');
      
      console.log('\n📊 Differenze tra le immagini:');
      console.log('═'.repeat(60));
      
      let differences = 0;
      for (let i = 0; i < img1.length; i++) {
        const qa1 = img1[i];
        const qa2 = img2[i];
        
        if (qa1 && qa2 && qa1.answer !== qa2.answer) {
          console.log(`Q${i}: A=${qa1.answer ? 'Sì' : 'No'} (${qa1.percentage}%) vs B=${qa2.answer ? 'Sì' : 'No'} (${qa2.percentage}%)`);
          differences++;
        }
      }
      
      console.log(`\n📈 Totale differenze: ${differences}/${img1.length}`);
      
      setResults([
        { imageName: 'Persona A', answers: img1, timestamp: Date.now() },
        { imageName: 'Persona B', answers: img2, timestamp: Date.now() + 1 }
      ]);
      
      setCurrentExample(`✅ ESEMPIO 6 completato (${differences} differenze)`);
      
    } catch (error) {
      console.error('❌ Errore:', error);
      setCurrentExample(`❌ Errore ESEMPIO 6: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Helper per ottenere il nome dell'attributo
   */
  const getAttributeName = (questionId: number): string => {
    const map: Record<number, string> = {
      0: 'Sorriso', 1: 'Uomo', 2: 'Donna',
      3: 'Capelli Marroni', 4: 'Capelli Neri', 5: 'Capelli Biondi', 6: 'Capelli Grigi',
      7: 'Capelli Lunghi', 8: 'Capelli Corti',
      9: 'Asiatico', 10: 'Nero', 11: 'Latino', 12: 'Bianco',
      13: 'Occhi Azzurri', 14: 'Occhi Marroni', 15: 'Occhi Verdi',
      16: 'Con Barba', 17: 'Con Occhiali',
      18: 'Nome Vocale', 19: 'Nome Consonante'
    };
    return map[questionId] || `Q${questionId}`;
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif', maxWidth: '1200px', margin: '0 auto' }}>
      <h1>🧠 ONNX Image Classifier - Esempi</h1>
      
      {/* Sezione inizializzazione */}
      <div style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
        <h2>Inizializzazione</h2>
        <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
          <button 
            onClick={() => initializeClassifier(true)} 
            disabled={isLoading}
            style={{ padding: '10px 20px', cursor: 'pointer' }}
          >
            Carica Modello (WebGPU)
          </button>
          <button 
            onClick={() => initializeClassifier(false)} 
            disabled={isLoading}
            style={{ padding: '10px 20px', cursor: 'pointer' }}
          >
            Carica Modello (WASM)
          </button>
        </div>
        {modelLoaded && (
          <div style={{ color: 'green', fontWeight: 'bold' }}>
            ✅ Modello caricato - Provider: {executionProvider}
          </div>
        )}
      </div>

      {/* Sezione esempi */}
      <div style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
        <h2>Esempi</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
          <button 
            onClick={example1_SingleImage} 
            disabled={!modelLoaded || isLoading}
            style={{ padding: '10px', cursor: modelLoaded ? 'pointer' : 'not-allowed' }}
          >
            1️⃣ Singola Immagine
          </button>
          <button 
            onClick={example2_ClassifyById} 
            disabled={!modelLoaded || isLoading}
            style={{ padding: '10px', cursor: modelLoaded ? 'pointer' : 'not-allowed' }}
          >
            2️⃣ Per ID
          </button>
          <button 
            onClick={example3_BatchProcessing} 
            disabled={!modelLoaded || isLoading}
            style={{ padding: '10px', cursor: modelLoaded ? 'pointer' : 'not-allowed' }}
          >
            3️⃣ Batch (5 img)
          </button>
          <button 
            onClick={example4_HighConfidence} 
            disabled={!modelLoaded || isLoading}
            style={{ padding: '10px', cursor: modelLoaded ? 'pointer' : 'not-allowed' }}
          >
            4️⃣ Alta Confidenza
          </button>
          <button 
            onClick={example5_DetailedAnalysis} 
            disabled={!modelLoaded || isLoading}
            style={{ padding: '10px', cursor: modelLoaded ? 'pointer' : 'not-allowed' }}
          >
            5️⃣ Analisi Dettagliata
          </button>
          <button 
            onClick={example6_CompareImages} 
            disabled={!modelLoaded || isLoading}
            style={{ padding: '10px', cursor: modelLoaded ? 'pointer' : 'not-allowed' }}
          >
            6️⃣ Comparazione
          </button>
        </div>
      </div>

      {/* Stato corrente */}
      {currentExample && (
        <div style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#e3f2fd', borderRadius: '8px' }}>
          <strong>Stato:</strong> {currentExample}
          {isLoading && <span> ⏳</span>}
        </div>
      )}

      {/* Risultati */}
      {results.length > 0 && (
        <div style={{ marginTop: '20px' }}>
          <h2>Risultati ({results.length})</h2>
          {results.map((result, idx) => (
            <div 
              key={result.timestamp} 
              style={{ 
                marginBottom: '20px', 
                padding: '15px', 
                border: '1px solid #ddd', 
                borderRadius: '8px',
                backgroundColor: '#fff'
              }}
            >
              <h3>📸 {result.imageName}</h3>
              <div style={{ fontSize: '12px', color: '#666', marginBottom: '10px' }}>
                Timestamp: {new Date(result.timestamp).toLocaleTimeString()}
              </div>
              
              {/* Tabella risultati */}
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f5f5f5' }}>
                    <th style={{ padding: '8px', textAlign: 'left', border: '1px solid #ddd' }}>ID</th>
                    <th style={{ padding: '8px', textAlign: 'left', border: '1px solid #ddd' }}>Attributo</th>
                    <th style={{ padding: '8px', textAlign: 'center', border: '1px solid #ddd' }}>Risposta</th>
                    <th style={{ padding: '8px', textAlign: 'right', border: '1px solid #ddd' }}>Confidenza</th>
                  </tr>
                </thead>
                <tbody>
                  {result.answers.map((qa) => (
                    <tr key={qa.questionId}>
                      <td style={{ padding: '8px', border: '1px solid #ddd' }}>{qa.questionId}</td>
                      <td style={{ padding: '8px', border: '1px solid #ddd' }}>{getAttributeName(qa.questionId)}</td>
                      <td style={{ 
                        padding: '8px', 
                        textAlign: 'center', 
                        border: '1px solid #ddd',
                        color: qa.answer ? 'green' : 'red',
                        fontWeight: 'bold'
                      }}>
                        {qa.answer ? '✓ Sì' : '✗ No'}
                      </td>
                      <td style={{ 
                        padding: '8px', 
                        textAlign: 'right', 
                        border: '1px solid #ddd',
                        backgroundColor: qa.percentage > 70 ? '#e8f5e9' : qa.percentage > 50 ? '#fff9c4' : '#ffebee'
                      }}>
                        {qa.percentage}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              
              {/* Statistiche rapide */}
              <div style={{ marginTop: '10px', fontSize: '14px', color: '#666' }}>
                <strong>Stats:</strong> {result.answers.filter(qa => qa.answer).length}/{result.answers.length} risposte positive
                {' • '}
                {result.answers.filter(qa => qa.percentage > 70).length} alta confidenza (&gt;70%)
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Footer con istruzioni */}
      <div style={{ marginTop: '40px', padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '8px', fontSize: '14px' }}>
        <h3>💡 Come usare:</h3>
        <ol>
          <li>Clicca su "Carica Modello" (WebGPU o WASM)</li>
          <li>Seleziona uno degli esempi da eseguire</li>
          <li>I risultati verranno mostrati nella sezione "Risultati"</li>
          <li>Controlla anche la console del browser per output dettagliato</li>
        </ol>
        <p><strong>Nota:</strong> Tutti i risultati sono anche stampati nella console del browser con formattazione dettagliata.</p>
      </div>
    </div>
  );
}

export default App;