// Main OCR GUI Component
const OCRQualityGUI = () => {
    const [pages, setPages] = useState([]);
    const [currentPage, setCurrentPage] = useState(0);
    const [loading, setLoading] = useState(true);
    const [reloading, setReloading] = useState(false);
    const [showErrors, setShowErrors] = useState(true);
    const [borderThickness, setBorderThickness] = useState(4);
    const [analysisMethod, setAnalysisMethod] = useState('supervised');
    const [viewType, setViewType] = useState('image'); // 'image', 'text', 'text_regions_gt', 'error_overlay_pred'
    const [sortOrder, setSortOrder] = useState('original');
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false);
    
    const availableMethods = [
        {
            id: 'chunks',
            name: 'Chunks Analysis', 
            description: 'Analyzes text segments and contextual patterns',
            available: true
        },
        {
            id: 'supervised',
            name: 'Supervised Approach',
            description: 'Builds a regression head on DistilBERT to predict error counts.',
            available: true  
        },
        {
            id: 'likelihood',
            name: 'Likelihood Analysis',
            description: 'Based on character-level probability scores',
            available: true
        }
    ];
    
    const loadPages = useCallback(async () => {
        try {
            console.log(`Loading pages with method: ${analysisMethod}`);
            const response = await fetch(`/api/pages?method=${analysisMethod}`);
            const data = await response.json();

            setPages(data);
            setLoading(false);
            console.log(`Loaded ${data.length} pages with ${analysisMethod} analysis`);
        } catch (error) {
            console.error('Error fetching pages:', error);
            setLoading(false);
        }
    }, [analysisMethod]);

    useEffect(() => {
        loadPages();
    }, [loadPages]);

    useEffect(() => {
        if (analysisMethod === 'supervised' && viewType !== 'image') {
            setViewType('image');
        }
        // For chunks and likelihood, switch to image view if current view doesn't have data
        else if ((analysisMethod === 'chunks' || analysisMethod === 'likelihood')) {
            // Get current page data to check what's available
            if (pages.length > 0 && currentPage < pages.length) {
                const currentPageData = pages[currentPage];
                
                // If current view is not available, switch to image view
                if (viewType === 'text' && !currentPageData.hasOcrText) {
                    setViewType('image');
                } else if (viewType === 'text_regions_gt' && !currentPageData.hasTextRegionsGT) {
                    setViewType('image');
                } else if (viewType === 'error_overlay_pred' && !currentPageData.hasErrorOverlayPred) {
                    setViewType('image');
                }
            }
        }
    }, [analysisMethod, pages, currentPage, viewType]);

    const handleReload = async () => {
        setReloading(true);
        try {
            console.log(`Reloading with method: ${analysisMethod}`);
            const response = await fetch('/api/reload', { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ method: analysisMethod })
            });
            const result = await response.json();
            console.log('Reload result:', result);
            await loadPages();
            setCurrentPage(0);
        } catch (error) {
            console.error('Error reloading images:', error);
        } finally {
            setReloading(false);
        }
    };

    const handleRunEvaluation = () => {
        console.log(`Navigating to data set visualization with method: ${analysisMethod}`);
        window.location.href = `/data_set_visualization?method=${analysisMethod}`;
    };

    const handleMethodChange = async (newMethod) => {
        if (availableMethods.find(m => m.id === newMethod)?.available) {
            console.log(`Changing method from ${analysisMethod} to ${newMethod}`);
            setAnalysisMethod(newMethod);
            setLoading(true);
            setSortOrder('original');
            setCurrentPage(0); 
        }
    };

     const getScoreForSort = (p) =>
        
        (typeof p.quality === 'number' ? p.quality : p.stats?.overallScore) ?? 0;

    const getSortedPages = () => {
        if (!pages || pages.length === 0) return [];

        switch (sortOrder) {
            case 'best-to-worst':
            return [...pages].sort((a, b) => getScoreForSort(b) - getScoreForSort(a));
            case 'worst-to-best':
                return [...pages].sort((a, b) => getScoreForSort(a) - getScoreForSort(b));
            case 'original':
            default:
                return pages;
        }
    };

    const handleSortChange = (newSortOrder) => {
        const currentPageData = getSortedPages()[currentPage];
        const currentFilename = currentPageData?.filename;
        
        setSortOrder(newSortOrder);
        
        if (currentFilename) {
            setTimeout(() => {
                const sortedPages = getSortedPages();
                const newIndex = sortedPages.findIndex(page => page.filename === currentFilename);
                if (newIndex !== -1) {
                    setCurrentPage(newIndex);
                }
            }, 0);
        }
    };

    const handleViewTypeChange = (newViewType) => {
        setViewType(newViewType);
    };

    const handlePageSelect = (pageIndex) => {
        setCurrentPage(pageIndex);
    };

    if (loading) {
        return React.createElement('div', {
            className: 'min-h-screen bg-gray-100 flex items-center justify-center'
        }, React.createElement('div', { className: 'text-center' }, [
            React.createElement('div', { key: 'spinner', className: 'animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4' }),
            React.createElement('div', { key: 'text', className: 'text-xl text-gray-600' }, `Loading OCR data with ${analysisMethod} analysis...`)
        ]));
    }

    const sortedPages = getSortedPages();
    const currentPageData = sortedPages[currentPage];

    if (!currentPageData) {
        return React.createElement('div', {
            className: 'min-h-screen bg-gray-100 flex items-center justify-center'
        }, React.createElement('div', { className: 'text-center' }, [
            React.createElement('div', { key: 'text', className: 'text-xl text-gray-600' }, `No pages found for ${analysisMethod} analysis`)
        ]));
    }

    const nextPage = () => setCurrentPage((prev) => (prev + 1) % sortedPages.length);
    const prevPage = () => setCurrentPage((prev) => (prev - 1 + sortedPages.length) % sortedPages.length);

    // predictionErrors for statistics, displayErrors for image overlay
    const predictionErrors = currentPageData.predictionErrors ? currentPageData.predictionErrors.filter(error => error.type === 'prediction_error') : [];
    const displayErrors = currentPageData.displayErrors || [];

    const getViewInfo = (viewType) => {
        switch (viewType) {
            case 'image':
                return { icon: FileImage, title: `${currentPageData.filename}` };
            case 'text':
                return { icon: FileText, title: currentPageData.filename };
            case 'text_regions_gt':
                return { icon: Layers, title: `${currentPageData.filename} (${analysisMethod})` };
            case 'error_overlay_pred':
                return { icon: Target, title: `${currentPageData.filename} (${analysisMethod})`};
            default:
                return { icon: FileImage, title: currentPageData.filename };
        }
    };

    const viewInfo = getViewInfo(viewType);

    const renderViewContent = () => {
        switch (viewType) {
            case 'image':
                return React.createElement(ImageViewer, {
                    src: currentPageData.imageUrl,
                    alt: `Document page ${currentPageData.id}`,
                    displayErrors: displayErrors, 
                    showErrors: showErrors,
                    analysisMethod: analysisMethod
                });
            case 'text':
                return React.createElement(OCRTextViewer, {
                    ocrTextData: currentPageData.ocrText,
                    pageId: currentPageData.id,
                    analysisMethod: analysisMethod
                });
            case 'text_regions_gt':
                return React.createElement(PNGViewer, {
                    src: currentPageData.pngRepresentations?.text_regions_gt ? 
                        `/text_regions_gt/${analysisMethod}/${currentPageData.pngRepresentations.text_regions_gt}` : null,
                    alt: `Text Regions GT for ${currentPageData.filename}`,
                    title: 'Text Regions Ground Truth',
                    type: 'text_regions_gt',
                    analysisMethod: analysisMethod
                });
            case 'error_overlay_pred':
                return React.createElement(PNGViewer, {
                    src: currentPageData.pngRepresentations?.error_overlay_pred ? 
                        `/error_overlay_pred/${analysisMethod}/${currentPageData.pngRepresentations.error_overlay_pred}` : null,
                    alt: `Error Overlay Pred for ${currentPageData.filename}`,
                    title: 'Error Overlay Predictions',
                    type: 'error_overlay_pred',
                    analysisMethod: analysisMethod
                });
            default:
                return React.createElement('div', {}, 'Unknown view type');
        }
    };
    const handleOpenBooks = () => {
        window.location.href = '/books';
    };


    // Main layout with header and left panel
    return React.createElement(React.Fragment, {}, [
        // Main application container
        React.createElement('div', { key: 'app', className: 'min-h-screen bg-gray-200' }, [
            // Header
            React.createElement(AppHeader, {
                key: 'header',
                onRunEvaluation: handleRunEvaluation,
                viewType: viewType,
                onViewTypeChange: handleViewTypeChange,
                hasOcrText: currentPageData?.hasOcrText,
                hasTextRegionsGT: currentPageData?.hasTextRegionsGT,
                hasErrorOverlayPred: currentPageData?.hasErrorOverlayPred,
                analysisMethod: analysisMethod,
                onOpenBooks: handleOpenBooks   
            }),
            // Main layout container
            React.createElement('div', { key: 'main', className: 'main-layout' }, [
                // Left sidebar with pages
                React.createElement(PagesSidebar, {
                    key: 'sidebar',
                    pages: sortedPages,
                    currentPage: currentPage,
                    onPageSelect: handlePageSelect,
                    analysisMethod: analysisMethod,
                    sortOrder: sortOrder,
                    onSortChange: handleSortChange,
                    isCollapsed: sidebarCollapsed,
                    onToggleCollapse: () => setSidebarCollapsed(!sidebarCollapsed)
                }),

                // Main content
                React.createElement('div', { key: 'content', className: `content-with-sidebar bg-gray-200 ${sidebarCollapsed ? 'sidebar-collapsed' : ''}` }, [
                    React.createElement('div', { className: 'max-w-7xl mx-auto p-6 h-full' }, [

                        React.createElement('div', { key: 'controls', className: 'mb-1 flex justify-end gap-2' }, [
                            React.createElement(MethodDropdown, {
                                key: 'method-dropdown',
                                selectedMethod: analysisMethod,
                                onMethodChange: handleMethodChange,
                                availableMethods: availableMethods
                            }),
                            
                            React.createElement(SortDropdown, {
                                key: 'sort-dropdown',
                                sortOrder: sortOrder,
                                onSortChange: handleSortChange
                            }),
                            
                            // Hide Lines button
                            React.createElement('button', {
                                key: 'toggle-errors',
                                onClick: () => setShowErrors(!showErrors),
                                disabled: viewType !== 'image',
                                className: `px-3 py-1 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                                    viewType !== 'image' ? 'bg-gray-400 text-gray-200 cursor-not-allowed' :
                                    showErrors ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-gray-600 text-white hover:bg-gray-700'
                                }`,
                                title: viewType !== 'image' ? 'Line overlays only available in image view' : 
                                       showErrors ? 'Hide all line overlays' : 'Show all line overlays'
                            }, [
                                React.createElement(showErrors ? EyeOff : Eye, { key: 'icon', className: 'w-4 h-4' }),
                                React.createElement('span', { key: 'text' }, showErrors ? 'Hide Lines' : 'Show Lines')
                            ])
                        ]),
                        
                        // Main content
                        React.createElement('div', { key: 'content', className: 'main-content-area h-[calc(100vh-200px)]' }, [
                            // Left Panel 
                            React.createElement('div', { 
                                key: 'left-panel', 
                                className: 'middle-content bg-white rounded-lg shadow-lg p-6 transition-all duration-500 mr-6'
                            }, [
                                React.createElement('div', { key: 'content-header', className: 'flex items-center justify-between mb-4' }, [
                                    React.createElement('h2', { key: 'title', className: 'text-xl font-semibold text-gray-800 flex items-center' }, [
                                        React.createElement(viewInfo.icon, { key: 'icon' }),
                                        React.createElement('span', { key: 'text', className: 'ml-2' }, viewInfo.title)
                                    ]),
                                    React.createElement('div', { key: 'nav-controls', className: 'flex items-center space-x-2' }, [
                                        React.createElement('button', {
                                            key: 'prev',
                                            onClick: prevPage,
                                            className: 'p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors',
                                            disabled: currentPage === 0
                                        }, React.createElement(ChevronLeft)),
                                        React.createElement('span', { key: 'counter', className: 'text-sm text-gray-600' }, 
                                            `${currentPage + 1} / ${sortedPages.length}`
                                        ),
                                        React.createElement('button', {
                                            key: 'next',
                                            onClick: nextPage,
                                            className: 'p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors',
                                            disabled: currentPage === sortedPages.length - 1
                                        }, React.createElement(ChevronRight))
                                    ])
                                ]),
                                React.createElement('div', { key: 'content-container', className: 'image-container border-2 border-gray-200 rounded-lg overflow-hidden bg-white h-[calc(100%-80px)]' },
                                    renderViewContent()
                                )
                            ]),

                            !rightPanelCollapsed && React.createElement(RightPanel, {
                                key: 'right-panel',
                                analysisMethod: analysisMethod,
                                currentPageData: currentPageData,
                                isCollapsed: rightPanelCollapsed,
                                onToggleCollapse: () => setRightPanelCollapsed(!rightPanelCollapsed),
                                availableMethods: availableMethods
                            })
                        ])
                    ])
                ])
            ])
        ]),

        rightPanelCollapsed && React.createElement(RightPanel, {
            key: 'right-panel-fixed',
            analysisMethod: analysisMethod,
            currentPageData: currentPageData,
            isCollapsed: rightPanelCollapsed,
            onToggleCollapse: () => setRightPanelCollapsed(!rightPanelCollapsed),
            availableMethods: availableMethods
        })
    ]);
};

ReactDOM.render(React.createElement(OCRQualityGUI), document.getElementById('root'));