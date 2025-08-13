// View Type Selector
const ViewTypeSelector = ({ viewType, onViewTypeChange, hasOcrText, hasTextRegionsGT, hasErrorOverlayPred, analysisMethod }) => {
    const isSupervised = analysisMethod === 'supervised';
    
    const isChunksOrLikelihood = analysisMethod === 'chunks' || analysisMethod === 'likelihood';
    
    const showOcrText = hasOcrText && !isSupervised;
    const showTextRegionsGT = hasTextRegionsGT && !isSupervised;
    const showErrorOverlay = hasErrorOverlayPred && !isSupervised;
    
    const availableButtons = [];
    
    // Image button is always available
    availableButtons.push(React.createElement('button', {
        key: 'image-view',
        onClick: () => onViewTypeChange('image'),
        className: `px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2 min-w-[120px] ${
            viewType === 'image' 
                ? 'bg-white text-gray-900 shadow-sm' 
                : 'text-gray-600 hover:text-gray-900'
        }`,
        title: isSupervised ? 'View document image (page-level prediction)' : 
               isChunksOrLikelihood ? 'View document image with error overlays' : 'View document image with all line overlays'
    }, [
        React.createElement(FileImage, { key: 'icon' }),
        React.createElement('span', { key: 'text' }, 'Image')
    ]));
    
    // OCR Text button only if available or supervised
    if (showOcrText || isSupervised) {
        availableButtons.push(React.createElement('button', {
            key: 'text-view',
            onClick: () => !isSupervised && hasOcrText && onViewTypeChange('text'),
            disabled: isSupervised || !hasOcrText,
            className: `px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2 min-w-[120px] ${
                viewType === 'text' && !isSupervised
                    ? 'bg-white text-gray-900 shadow-sm' 
                    : (isSupervised || !hasOcrText) ? 'text-gray-400 cursor-not-allowed' : 'text-gray-600 hover:text-gray-900'
            }`,
            title: isSupervised ? 'Not available for supervised approach (page-level predictions only)' :
                   hasOcrText ? 'View OCR text output' : 'No OCR text data available'
        }, [
            React.createElement(FileText, { key: 'icon' }),
            React.createElement('span', { key: 'text' }, 'OCR Text')
        ]));
    }
    
    // Text Regions GT button only if available or supervised
    if (showTextRegionsGT || isSupervised) {
        availableButtons.push(React.createElement('button', {
            key: 'text-regions-gt',
            onClick: () => !isSupervised && hasTextRegionsGT && onViewTypeChange('text_regions_gt'),
            disabled: isSupervised || !hasTextRegionsGT,
            className: `px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2 min-w-[140px] ${
                viewType === 'text_regions_gt' && !isSupervised
                    ? 'bg-white text-gray-900 shadow-sm' 
                    : (isSupervised || !hasTextRegionsGT) ? 'text-gray-400 cursor-not-allowed' : 'text-gray-600 hover:text-gray-900'
            }`,
            title: isSupervised ? 'Not available for supervised approach (page-level predictions only)' :
                   hasTextRegionsGT ? 'View text regions ground truth' : 'No text regions GT available'
        }, [
            React.createElement(Layers, { key: 'icon' }),
            React.createElement('span', { key: 'text' }, 'Text Regions GT')
        ]));
    }
    
    // Error Overlay button only if available or supervised
    if (showErrorOverlay || isSupervised) {
        availableButtons.push(React.createElement('button', {
            key: 'error-overlay-pred',
            onClick: () => !isSupervised && hasErrorOverlayPred && onViewTypeChange('error_overlay_pred'),
            disabled: isSupervised || !hasErrorOverlayPred,
            className: `px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2 min-w-[160px] ${
                viewType === 'error_overlay_pred' && !isSupervised
                    ? 'bg-white text-gray-900 shadow-sm' 
                    : (isSupervised || !hasErrorOverlayPred) ? 'text-gray-400 cursor-not-allowed' : 'text-gray-600 hover:text-gray-900'
            }`,
            title: isSupervised ? 'Not available for supervised approach (page-level predictions only)' :
                   hasErrorOverlayPred ? 'View error overlay predictions' : 'No error overlay predictions available'
        }, [
            React.createElement(Target, { key: 'icon' }),
            React.createElement('span', { key: 'text' }, 'Error Overlay Pred')
        ]));
    }
    
    return React.createElement('div', { 
        className: 'flex bg-gray-100 rounded-lg p-1 gap-1' 
    }, availableButtons);
};