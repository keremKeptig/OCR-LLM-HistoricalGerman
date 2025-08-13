// Image Viewer 
const ImageViewer = ({ src, alt, displayErrors, showErrors, borderThickness, analysisMethod }) => {
    const [scale, setScale] = useState(1);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
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

    return React.createElement('div', {
        ref: containerRef,
        className: 'image-container w-full h-full relative',
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
        React.createElement('img', {
            key: 'image',
            ref: imageRef,
            src: src,
            alt: alt,
            className: 'image-viewer w-full h-full object-contain',
            style: createTransformStyle(scale, position.x, position.y),
            onMouseDown: handleMouseDown,
            draggable: false
        }),
        React.createElement(ErrorOverlay, {
            key: 'overlay',
            displayErrors: displayErrors, 
            imageRef: imageRef,
            scale: scale,
            position: position,
            showErrors: showErrors,
            borderThickness: borderThickness,
            analysisMethod: analysisMethod
        })
    ]);
};