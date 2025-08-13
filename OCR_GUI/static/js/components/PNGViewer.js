// PNG Viewer Component 
const PNGViewer = ({ src, alt, title, type, analysisMethod }) => {
    const [scale, setScale] = useState(1);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(false);
    const containerRef = useRef(null);
    const imageRef = useRef(null);

    const zoomIn = () => setScale(prev => Math.min(prev * 1.2, 5));
    const zoomOut = () => setScale(prev => Math.max(prev / 1.2, 0.1));
    const resetZoom = () => {
        setScale(1);
        setPosition({ x: 0, y: 0 });
    };

    const handleMouseDown = (e) => {
        if (scale > 1) {
            setIsDragging(true);
            setDragStart({
                x: e.clientX - position.x,
                y: e.clientY - position.y
            });
        }
    };

    const handleMouseMove = (e) => {
        if (isDragging && scale > 1) {
            setPosition({
                x: e.clientX - dragStart.x,
                y: e.clientY - dragStart.y
            });
        }
    };

    const handleMouseUp = () => setIsDragging(false);

    const handleWheel = (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        setScale(prev => Math.max(0.1, Math.min(5, prev * delta)));
    };

    useEffect(() => {
        if (isDragging) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            return () => {
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
            };
        }
    }, [isDragging, dragStart]);

    if (error || !src) {
        return React.createElement('div', {
            className: 'flex flex-col items-center justify-center h-full text-gray-500'
        }, [
            React.createElement(type === 'text_regions_gt' ? Layers : Target, { 
                key: 'icon', 
                className: 'w-16 h-16 mb-4' 
            }),
            React.createElement('div', { key: 'title', className: 'text-xl font-medium mb-2' }, 
                `No ${title} Available`
            ),
            React.createElement('div', { key: 'message', className: 'text-sm text-center max-w-sm' }, 
                `${title} file is not available for this document using ${analysisMethod} analysis. Make sure PNG files are placed in the correct method-specific folder.`
            )
        ]);
    }

    return React.createElement('div', {
        ref: containerRef,
        className: 'png-viewer-container w-full h-full relative',
        onWheel: handleWheel
    }, [
        React.createElement('div', { key: 'controls', className: 'zoom-controls' }, [
            React.createElement('button', {
                key: 'zoomin',
                onClick: zoomIn,
                className: 'p-2 bg-white shadow-lg rounded-lg hover:bg-gray-50 transition-colors',
                title: 'Zoom In'
            }, React.createElement(ZoomIn)),
            React.createElement('button', {
                key: 'zoomout',
                onClick: zoomOut,
                className: 'p-2 bg-white shadow-lg rounded-lg hover:bg-gray-50 transition-colors',
                title: 'Zoom Out'
            }, React.createElement(ZoomOut)),
            React.createElement('button', {
                key: 'reset',
                onClick: resetZoom,
                className: 'p-2 bg-white shadow-lg rounded-lg hover:bg-gray-50 transition-colors',
                title: 'Reset Zoom'
            }, React.createElement(RotateCw)),
            React.createElement('div', {
                key: 'indicator',
                className: 'text-xs bg-white px-2 py-1 rounded shadow-lg'
            }, Math.round(scale * 100) + '%')
        ]),
        loading && React.createElement('div', {
            key: 'loading',
            className: 'absolute inset-0 flex items-center justify-center bg-gray-50'
        }, [
            React.createElement('div', { key: 'spinner', className: 'animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600' }),
            React.createElement('span', { key: 'text', className: 'ml-3 text-gray-600' }, `Loading ${title} (${analysisMethod})...`)
        ]),
        React.createElement('img', {
            key: 'image',
            ref: imageRef,
            src: src,
            alt: alt,
            className: 'png-viewer w-full h-full object-contain',
            style: createTransformStyle(scale, position.x, position.y),
            onMouseDown: handleMouseDown,
            onLoad: () => setLoading(false),
            onError: () => {
                setLoading(false);
                setError(true);
            },
            draggable: false
        })
    ]);
};