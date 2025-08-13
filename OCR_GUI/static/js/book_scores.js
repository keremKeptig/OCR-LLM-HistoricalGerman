const {useState,useEffect,useMemo,useRef} = React;

function NumberPill({label, value}) {
  return (
    <div className="flex flex-col items-center justify-center p-4 bg-white rounded-xl shadow-md">
      <div className="text-sm text-gray-500">{label}</div>
      <div className="text-xl font-semibold">{value}</div>
    </div>
  );
}

function BookScoresApp(){
  const [summaries,setSummaries] = useState([]);
  const [selected,setSelected] = useState('');
  const [pages,setPages] = useState([]);
  const [loading,setLoading] = useState(true);
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(()=>{
    (async ()=>{
      const res = await fetch('/api/books/summaries');
      const data = await res.json();
      setSummaries(data);
      if (data.length > 0) setSelected(data[0].book_id);
      setLoading(false);
    })();
  },[]);

  useEffect(()=>{
    if (!selected) return;
    (async ()=>{
      const res = await fetch(`/api/books/page-scores?book_id=${encodeURIComponent(selected)}&sort=asc`);
      const data = await res.json();
      setPages(data);
    })();
  },[selected]);

  // draw chart
  useEffect(()=>{
    if (!chartRef.current) return;
    if (chartInstance.current) {
      chartInstance.current.destroy();
      chartInstance.current = null;
    }
    if (pages.length === 0) return;

    const labels = pages.map(p => p.page_id);
    const values = pages.map(p => p.score);
    const ctx = chartRef.current.getContext('2d');

    chartInstance.current = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Predicted error rate (lower = better)',
          data: values
        }]
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            ticks: { maxRotation: 75, minRotation: 75, autoSkip: true, maxTicksLimit: 40 },
            title: { display: true, text: 'Page ID' }
          },
          y: {
            min: 0, max: 1,
            title: { display: true, text: 'Predicted error rate' }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(4)}` } },
          title: {
            display: true,
            text: `${selected}: predicted per-page scores (sorted)`
          }
        }
      }
    });
  },[pages,selected]);

  const selectedSummary = useMemo(
    ()=> summaries.find(s => s.book_id === selected),
    [summaries, selected]
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-600">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mr-3"></div>
        Loading…
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* controls */}
      <div className="flex flex-wrap items-center gap-3">
        <label className="text-sm text-gray-600">Book</label>
        <select
          value={selected}
          onChange={e=>setSelected(e.target.value)}
          className="px-3 py-2 border rounded-lg bg-white"
        >
          {summaries.map(s=>(
            <option key={s.book_id} value={s.book_id}>{s.book_id}</option>
          ))}
        </select>
        <button
          onClick={()=>{
            const csvRows = [['book_id','page_id','xml_path','score'], ...pages.map(p=>[p.book_id,p.page_id,p.xml_path,p.score])];
            const blob = new Blob([csvRows.map(r=>r.join(',')).join('\n')],{type:'text/csv'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${selected}_page_scores.csv`;
            a.click();
            URL.revokeObjectURL(url);
          }}
          className="px-3 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700"
        >
          Download page scores CSV
        </button>
      </div>

      {/* summary cards */}
      {selectedSummary && (
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
          <NumberPill label="Pages" value={selectedSummary.num_pages} />
          <NumberPill label="Mean Error" value={(+selectedSummary.mean_score * 100).toFixed(2) + '%'} />
          <NumberPill label="Median Error" value={(+selectedSummary.median_score * 100).toFixed(2) + '%'} />
          <NumberPill label="Best Score" value={(+selectedSummary.best_score * 100).toFixed(2) + '%'} />
          <NumberPill label="Worst Score" value={(+selectedSummary.worst_score * 100).toFixed(2) + '%'} />
        </div>
      )}

      {/* chart */}
      <div className="bg-white rounded-xl shadow-md p-4 h-[520px]">
        <canvas ref={chartRef} className="w-full h-full"></canvas>
      </div>

      {/* table */}
      <div className="bg-white rounded-xl shadow-md overflow-hidden">
        <div className="px-4 py-3 border-b font-medium">Per-page scores</div>
        <div className="overflow-auto max-h-[420px]">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                <th className="text-left px-4 py-2">Page ID</th>
                <th className="text-left px-4 py-2">XML Path</th>
                <th className="text-right px-4 py-2">Score</th>
              </tr>
            </thead>
            <tbody>
              {pages.map((p,i)=>(
                <tr key={i} className={i%2? 'bg-white':'bg-gray-50'}>
                  <td className="px-4 py-2 font-mono">{p.page_id}</td>
                  <td className="px-4 py-2 whitespace-nowrap">{p.xml_path}</td>
                  <td className="px-4 py-2 text-right">{(+p.score).toFixed(6)}</td>
                </tr>
              ))}
              {pages.length===0 && (
                <tr><td colSpan="3" className="px-4 py-6 text-center text-gray-500">No page scores</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('books-root')).render(<BookScoresApp />);
