// OCR Text Viewer Component
const OCRTextViewer = ({ ocrTextData, pageId, analysisMethod }) => {
    const [textData, setTextData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchOCRText = async () => {
            if (ocrTextData && ocrTextData.ColoredTextHTML) {
                setTextData(ocrTextData);
                setLoading(false);
                return;
            }

            try {
                setLoading(true);
                const response = await fetch(`/api/page/${pageId}/ocr-text?method=${analysisMethod}`);
                
                if (!response.ok) {
                    throw new Error('OCR text data not available');
                }
                
                const data = await response.json();
                setTextData(data);
                setError(null);
            } catch (err) {
                setError(err.message);
                setTextData(null);
            } finally {
                setLoading(false);
            }
        };

        fetchOCRText();
    }, [pageId, analysisMethod, ocrTextData]);

    if (loading) {
        return React.createElement('div', {
            className: 'flex items-center justify-center h-full'
        }, [
            React.createElement('div', { key: 'spinner', className: 'animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600' }),
            React.createElement('span', { key: 'text', className: 'ml-3 text-gray-600' }, 'Loading OCR text...')
        ]);
    }

    if (error || !textData) {
        return React.createElement('div', {
            className: 'flex flex-col items-center justify-center h-full text-gray-500'
        }, [
            React.createElement(FileText, { key: 'icon', className: 'w-16 h-16 mb-4' }),
            React.createElement('div', { key: 'title', className: 'text-xl font-medium mb-2' }, 'No OCR Text Available'),
            React.createElement('div', { key: 'message', className: 'text-sm text-center max-w-sm' }, 
                error || 'OCR text data is not available for this document. Make sure OCR text files are placed in the ocr_text folder.'
            )
        ]);
    }

    return React.createElement('div', {
        className: 'h-full overflow-auto p-6 bg-white'
    }, [
        React.createElement('div', {
            key: 'text-content',
            className: 'prose prose-sm max-w-none',
            style: {
                fontFamily: 'Georgia, serif',
                lineHeight: '1.8',
                fontSize: '16px'
            },
            dangerouslySetInnerHTML: { __html: textData.ColoredTextHTML }
        }),
        React.createElement('div', {
            key: 'legend',
            className: 'mt-8 p-4 bg-gray-50 rounded-lg border'
        }, [
            React.createElement('h4', { key: 'title', className: 'text-sm font-semibold text-gray-800 mb-3' }, 'OCR Confidence Legend'),
            React.createElement('div', { key: 'legend-items', className: 'grid grid-cols-2 gap-2 text-xs' }, [
                React.createElement('div', { key: 'high', className: 'flex items-center gap-2' }, [
                    React.createElement('span', { className: 'w-4 h-4 rounded', style: { backgroundColor: 'lightgreen' } }),
                    React.createElement('span', {}, 'High Confidence')
                ]),
                React.createElement('div', { key: 'medium', className: 'flex items-center gap-2' }, [
                    React.createElement('span', { className: 'w-4 h-4 rounded', style: { backgroundColor: 'orange' } }),
                    React.createElement('span', {}, 'Medium Confidence')
                ]),
                React.createElement('div', { key: 'low', className: 'flex items-center gap-2' }, [
                    React.createElement('span', { className: 'w-4 h-4 rounded', style: { backgroundColor: 'red' } }),
                    React.createElement('span', {}, 'Low Confidence')
                ])
            ])
        ])
    ]);
};