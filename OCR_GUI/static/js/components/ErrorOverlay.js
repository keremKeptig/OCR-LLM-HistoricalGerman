// Error Overlay Component
const ErrorOverlay = ({ displayErrors, imageRef, scale, position, showErrors, borderThickness = 0.4, analysisMethod }) => {
    const [tooltip, setTooltip] = useState(null);
    const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
    const [imageNaturalSize, setImageNaturalSize] = useState({ width: 0, height: 0 });
    
    const BORDER_THICKNESS = 0.4;  
    const HOVER_THICKNESS = borderThickness; 
    const BACKGROUND_OPACITY = '10'; 

    useEffect(() => {
        if (!imageRef.current) return;

        const updateDimensions = () => {
            const img = imageRef.current;
            const container = img.parentElement;
            
            const containerRect = container.getBoundingClientRect();
            setContainerSize({ 
                width: containerRect.width, 
                height: containerRect.height 
            });

            if (img.naturalWidth && img.naturalHeight) {
                setImageNaturalSize({ 
                    width: img.naturalWidth, 
                    height: img.naturalHeight 
                });
            }
        };
        
        updateDimensions();
        
        const img = imageRef.current;
        img.addEventListener('load', updateDimensions);
        window.addEventListener('resize', updateDimensions);
        
        return () => {
            img.removeEventListener('load', updateDimensions);
            window.removeEventListener('resize', updateDimensions);
        };
    }, [imageRef]);

    if (!showErrors || !displayErrors || displayErrors.length === 0 || !containerSize.width || !imageNaturalSize.width) {
        return null;
    }

    const filteredErrors = displayErrors.filter(error => {
        return error.type === 'line_error' && 
               error.line_id && 
               error.coords && 
               Array.isArray(error.coords) && 
               error.coords.length >= 4 &&
               error.method === 'xml_all_lines';
    });

    if (filteredErrors.length === 0) {
        console.log(`No display lines found from XML`);
        return null;
    }

    console.log(`Displaying ${filteredErrors.length} ALL lines from XML`);

    const imageAspectRatio = imageNaturalSize.width / imageNaturalSize.height;
    const containerAspectRatio = containerSize.width / containerSize.height;
    
    let displayedImageWidth, displayedImageHeight, offsetX = 0, offsetY = 0;
    
    if (imageAspectRatio > containerAspectRatio) {
        displayedImageWidth = containerSize.width;
        displayedImageHeight = containerSize.width / imageAspectRatio;
        offsetY = (containerSize.height - displayedImageHeight) / 2;
    } else {
        displayedImageHeight = containerSize.height;
        displayedImageWidth = containerSize.height * imageAspectRatio;
        offsetX = (containerSize.width - displayedImageWidth) / 2;
    }

    const scaleX = displayedImageWidth / imageNaturalSize.width;
    const scaleY = displayedImageHeight / imageNaturalSize.height;

    const getLineColor = () => '#3b82f6'; 

    return React.createElement('div', {
        className: 'error-overlay',
        style: {
            transform: `scale(${scale}) translate(${position.x / scale}px, ${position.y / scale}px)`,
            transformOrigin: '0 0'
        }
    }, [
        ...filteredErrors.map((error, index) => {
            const coords = error.coords;
            
            const xs = coords.map(coord => coord[0]);
            const ys = coords.map(coord => coord[1]);
            const minX = Math.min(...xs);
            const maxX = Math.max(...xs);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);

            const left = offsetX + (minX * scaleX);
            const top = offsetY + (minY * scaleY);
            const width = (maxX - minX) * scaleX;
            const height = (maxY - minY) * scaleY;

            const lineColor = getLineColor();

            return React.createElement('div', {
                key: `display-line-${error.line_id}-${index}`,
                className: 'line-display-box',
                style: {
                    position: 'absolute',
                    left: `${left}px`,
                    top: `${top}px`,
                    width: `${width}px`,
                    height: `${height}px`,
                    border: `${BORDER_THICKNESS}px solid ${lineColor}`,
                    backgroundColor: `${lineColor}${BACKGROUND_OPACITY}`, 
                    pointerEvents: 'auto',
                    cursor: 'pointer',
                    boxSizing: 'border-box',
                    borderRadius: '2px'
                },
                onMouseEnter: (e) => {
                    const rect = e.target.getBoundingClientRect();
                    setTooltip({
                        error,
                        x: rect.left + rect.width / 2,
                        y: rect.top - 10
                    });
                    
                    e.target.style.border = `${HOVER_THICKNESS}px solid ${lineColor}`;
                    e.target.style.backgroundColor = `${lineColor}30`;
                    e.target.style.boxShadow = `0 0 0 2px ${lineColor}40`;
                },
                onMouseLeave: (e) => {
                    setTooltip(null);
                    e.target.style.border = `${BORDER_THICKNESS}px solid ${lineColor}`;
                    e.target.style.backgroundColor = `${lineColor}${BACKGROUND_OPACITY}`;
                    e.target.style.boxShadow = 'none';
                },
                onClick: (e) => {
                    e.stopPropagation();
                    console.log(`Line clicked:`, {
                        line_id: error.line_id,
                        text: error.text
                    });
                }
            });
        }),
        
        tooltip && React.createElement('div', {
            key: 'tooltip',
            style: {
                position: 'fixed',
                left: `${tooltip.x}px`,
                top: `${tooltip.y}px`,
                transform: 'translateX(-50%) translateY(-100%)',
                background: '#1f2937',
                color: 'white',
                padding: '16px 20px',
                borderRadius: '8px',
                fontSize: '13px',
                fontFamily: 'system-ui, -apple-system, sans-serif',
                whiteSpace: 'nowrap',
                zIndex: 30,
                boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.3)',
                pointerEvents: 'none',
                maxWidth: '400px',
                minWidth: '250px'
            }
        }, [
            React.createElement('div', { 
                key: 'line-id', 
                style: { 
                    fontWeight: 'bold', 
                    fontSize: '16px',
                    marginBottom: '8px',
                    color: '#60a5fa',
                    textAlign: 'center'
                } 
            }, `Line: ${tooltip.error.line_id}`),
            
            React.createElement('div', { 
                key: 'line-text', 
                style: { 
                    fontSize: '12px',
                    marginBottom: '8px',
                    color: '#e5e7eb',
                    fontFamily: 'monospace',
                    whiteSpace: 'normal',
                    maxWidth: '350px',
                    wordBreak: 'break-word',
                    maxHeight: '60px',
                    overflow: 'hidden'
                } 
            }, `"${tooltip.error.text || 'No text available'}"`),
            
            React.createElement('div', { 
                key: 'info',
                style: { 
                    fontSize: '11px',
                    color: '#9ca3af',
                    textAlign: 'center',
                    borderTop: '1px solid #374151',
                    paddingTop: '6px'
                }
            }, 'Text line from XML document')
        ])
    ]);
};