// Right Panel Component
const RightPanel = ({ analysisMethod, currentPageData, isCollapsed, onToggleCollapse, availableMethods }) => {

    const getScoreColor = (score) => {
        if (score >= 0.9) return 'text-green-600';
        if (score >= 0.7) return 'text-yellow-600';
        return 'text-red-600';
    };

    const getScoreIcon = (score) => {
        if (score >= 0.9) return React.createElement(CheckCircle, { className: "w-5 h-5 text-green-600" });
        if (score >= 0.7) return React.createElement(AlertTriangle, { className: "w-5 h-5 text-yellow-600" });
        return React.createElement(AlertTriangle, { className: "w-5 h-5 text-red-600" });
    };

    const getReadabilityCategory = (score) => {
        if (score >= 0.9) return { text: 'Good', color: 'text-green-600', bgColor: 'bg-green-100' };
        if (score >= 0.7) return { text: 'Fair', color: 'text-yellow-600', bgColor: 'bg-yellow-100' };
        if (score >= 0.5) return { text: 'Poor', color: 'text-red-600', bgColor: 'bg-red-100' };
        return { text: 'Very Poor / Problematic', color: 'text-red-800', bgColor: 'bg-red-200' };
    };

    const displayErrors = currentPageData?.displayErrors || [];
    const predictionErrors = currentPageData?.predictionErrors?.filter(error => error.type === 'prediction_error') || [];

    // supervised right panel
    const renderSupervisedRightPanel = () => {
        return React.createElement('div', { key: 'right-panel', className: 'space-y-6 overflow-y-auto h-full' }, [
            // Overall Quality Score 
            React.createElement('div', { key: 'quality-score', className: 'bg-white rounded-lg shadow-lg p-6' }, [
                React.createElement('h3', { key: 'score-title', className: 'text-lg font-semibold text-gray-800 mb-4 flex items-center' }, [
                    getScoreIcon(currentPageData.stats.overallScore),
                    React.createElement('span', { key: 'text', className: 'ml-2' }, 'Overall Quality Score')
                ]),
                React.createElement('div', { key: 'score-content', className: 'text-center' }, [
                    React.createElement('div', { 
                        key: 'percentage',
                        className: `text-4xl font-bold ${getScoreColor(currentPageData.stats.overallScore)}`
                    }, (currentPageData.stats.overallScore * 100).toFixed(1) + '%'),
                    React.createElement('div', { 
                        key: 'readability',
                        className: `inline-flex items-center px-3 py-1 rounded-full text-sm font-medium mt-2 ${getReadabilityCategory(currentPageData.stats.overallScore).color} ${getReadabilityCategory(currentPageData.stats.overallScore).bgColor}`
                    }, getReadabilityCategory(currentPageData.stats.overallScore).text),
                    React.createElement('div', { key: 'progress-container', className: 'w-full bg-gray-200 rounded-full h-3 mt-3' },
                        React.createElement('div', {
                            className: `h-3 rounded-full ${
                                currentPageData.stats.overallScore >= 0.9 ? 'bg-green-600' :
                                currentPageData.stats.overallScore >= 0.7 ? 'bg-yellow-600' : 'bg-red-600'
                            }`,
                            style: createProgressStyle(currentPageData.stats.overallScore * 100)
                        })
                    )
                ])
            ]),

            // Supervised-specific metrics
            React.createElement('div', { key: 'metrics', className: 'bg-white rounded-lg shadow-lg p-6' }, [
                React.createElement('h3', { key: 'metrics-title', className: 'text-lg font-semibold text-gray-800 mb-4' }, 
                    'Supervised Analysis Metrics'
                ),
                React.createElement('div', { key: 'metrics-grid', className: 'grid grid-cols-2 gap-4' }, [
                    React.createElement('div', { key: 'words', className: 'text-center p-3 bg-green-50 rounded-lg' }, [
                        React.createElement('div', { key: 'value', className: 'text-2xl font-bold text-green-600' },
                            currentPageData.stats.wordCount
                        ),
                        React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 'Total Words')
                    ]),
                    React.createElement('div', { key: 'model-type', className: 'text-center p-3 bg-purple-50 rounded-lg' }, [
                       React.createElement('div', { key: 'value', className: 'text-lg font-bold text-purple-600' },
                            currentPageData.stats.pageErrorScore ? 
                                parseInt(currentPageData.stats.pageErrorScore * currentPageData.stats.wordCount)
                                : 'N/A'
                        ),
                        React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 'Predicted Word Errors')
                    ]),
                    React.createElement('div', { key: 'total-lines', className: 'text-center p-3 bg-blue-50 rounded-lg' }, [
                        React.createElement('div', { key: 'value', className: 'text-2xl font-bold text-blue-600' },
                            displayErrors.length
                        ),
                        React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 'Total Lines')
                    ]),
                    React.createElement('div', { key: 'error-score', className: 'text-center p-3 bg-red-50 rounded-lg' }, [
                        React.createElement('div', { key: 'value', className: 'text-2xl font-bold text-red-600' },
                            currentPageData.stats.pageErrorScore ? (currentPageData.stats.pageErrorScore * 100).toFixed(1) + '%' : 'N/A'
                        ),
                        React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 'Page Error Score')
                    ])
                ])
            ]),

            // Supervised method information
            React.createElement('div', { key: 'method-info', className: 'bg-white rounded-lg shadow-lg p-6' }, [
                React.createElement('h3', { key: 'info-title', className: 'text-lg font-semibold text-gray-800 mb-4' }, 
                    'Supervised Learning Approach'
                ),
                React.createElement('div', { key: 'info-content', className: 'space-y-4' }, [
                    React.createElement('div', { key: 'description', className: 'p-4 bg-purple-50 rounded-lg' }, [
                        React.createElement('h4', { key: 'desc-title', className: 'font-semibold text-purple-800 mb-2' }, 'Model Architecture'),
                        React.createElement('p', { key: 'desc-text', className: 'text-sm text-purple-700' }, 
                            'This approach uses a regression model trained on page embeddings and structural features to predict page-level error score.'
                        )
                    ]),
                    currentPageData.stats.pageErrorScore && React.createElement('div', { key: 'score-details', className: 'p-4 bg-gray-50 rounded-lg' }, [
                        React.createElement('h4', { key: 'score-title', className: 'font-semibold text-gray-800 mb-2' }, 'Prediction Details'),
                        React.createElement('div', { key: 'score-items', className: 'space-y-2 text-sm' }, [
                            React.createElement('div', { key: 'raw-score' }, [
                                React.createElement('span', { className: 'font-medium' }, 'Raw Error Score: '),
                                React.createElement('span', { className: 'font-mono text-red-600' }, 
                                    currentPageData.stats.pageErrorScore.toFixed(6)
                                )
                            ]),
                            React.createElement('div', { key: 'quality-score' }, [
                                React.createElement('span', { className: 'font-medium' }, 'Quality Score: '),
                                React.createElement('span', { className: 'font-mono text-green-600' }, 
                                    (currentPageData.stats.overallScore * 100).toFixed(1) + '%'
                                )
                            ]),
                            React.createElement('div', { key: 'prediction-type' }, [
                                React.createElement('span', { className: 'font-medium' }, 'Prediction Type: '),
                                React.createElement('span', { className: 'text-purple-600' }, 'Page-level error rate')
                            ])
                        ])
                    ])
                ])
            ])
        ]);
    };

    // right panel for likelihood and chunks
    const renderRegularRightPanel = () => {
        return React.createElement('div', { key: 'right-panel', className: 'space-y-6 overflow-y-auto h-full' }, [
            // Overall Quality Score
            React.createElement('div', { key: 'quality-score', className: 'bg-white rounded-lg shadow-lg p-6' }, [
                React.createElement('h3', { key: 'score-title', className: 'text-lg font-semibold text-gray-800 mb-4 flex items-center' }, [
                    getScoreIcon(currentPageData.stats.overallScore),
                    React.createElement('span', { key: 'text', className: 'ml-2' }, 'Overall Quality Score')
                ]),
                React.createElement('div', { key: 'score-content', className: 'text-center' }, [
                    React.createElement('div', { 
                        key: 'percentage',
                        className: `text-4xl font-bold ${getScoreColor(currentPageData.stats.overallScore)}`
                    }, (currentPageData.stats.overallScore * 100).toFixed(1) + '%'),
                    React.createElement('div', { 
                        key: 'readability',
                        className: `inline-flex items-center px-3 py-1 rounded-full text-sm font-medium mt-2 ${getReadabilityCategory(currentPageData.stats.overallScore).color} ${getReadabilityCategory(currentPageData.stats.overallScore).bgColor}`
                    }, getReadabilityCategory(currentPageData.stats.overallScore).text),
                    React.createElement('div', { key: 'progress-container', className: 'w-full bg-gray-200 rounded-full h-3 mt-3' },
                        React.createElement('div', {
                            className: `h-3 rounded-full ${
                                currentPageData.stats.overallScore >= 0.9 ? 'bg-green-600' :
                                currentPageData.stats.overallScore >= 0.7 ? 'bg-yellow-600' : 'bg-red-600'
                            }`,
                            style: createProgressStyle(currentPageData.stats.overallScore * 100)
                        })
                    )
                ])
            ]),

            // Key Metrics
            React.createElement('div', { key: 'metrics', className: 'bg-white rounded-lg shadow-lg p-6' }, [
                React.createElement('h3', { key: 'metrics-title', className: 'text-lg font-semibold text-gray-800 mb-4' }, 
                    `${availableMethods.find(m => m.id === analysisMethod)?.name || 'Analysis'} Metrics`
                ),
                React.createElement('div', { key: 'metrics-grid', className: 'grid grid-cols-2 gap-4' }, [
                    React.createElement('div', { key: 'words', className: 'text-center p-3 bg-green-50 rounded-lg' }, [
                        React.createElement('div', { key: 'value', className: 'text-2xl font-bold text-green-600' },
                            currentPageData.stats.wordCount
                        ),
                        React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 'Total Words')
                    ]),
                    React.createElement('div', { key: 'errors', className: 'text-center p-3 bg-orange-50 rounded-lg' }, [
                        React.createElement('div', { key: 'value', className: 'text-2xl font-bold text-orange-600' },
                            currentPageData.stats.errorCount
                        ),
                        React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 
                            analysisMethod === 'chunks' ? 'Predicted Words Errors' : 'Predicted Word Errors'
                        )
                    ]),
                    React.createElement('div', { key: 'total-lines', className: 'text-center p-3 bg-blue-50 rounded-lg' }, [
                        React.createElement('div', { key: 'value', className: 'text-2xl font-bold text-blue-600' },
                            displayErrors.length
                        ),
                        React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 'Total Lines (Display)')
                    ]),
                    React.createElement('div', { key: 'error-lines', className: 'text-center p-3 bg-red-50 rounded-lg' }, [
                        React.createElement('div', { key: 'value', className: 'text-2xl font-bold text-red-600' },
                            predictionErrors.length
                        ),
                        React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 'Lines with Predictions')
                    ])
                ])
            ]),

            // prediction Error Details
            React.createElement('div', { key: 'prediction-errors', className: 'bg-white rounded-lg shadow-lg p-6' }, [
                React.createElement('h3', { key: 'errors-title', className: 'text-lg font-semibold text-gray-800 mb-4' }, 
                    `${analysisMethod === 'chunks' ? 'Chunk' : 'Word'} Error Predictions`
                ),
                predictionErrors.length > 0 ? 
                    React.createElement('div', { key: 'errors-list', className: 'space-y-3 max-h-64 overflow-y-auto' }, [
                        ...predictionErrors.slice(0, predictionErrors.length).map((lineError, index) => 
                            React.createElement('div', { 
                                key: index, 
                                className: `p-4 rounded-lg ${
                                    lineError.coords ? 'bg-red-50 border border-red-200' : 'bg-gray-50'
                                }` 
                            }, [
                                React.createElement('div', { key: 'line-header', className: 'flex items-center justify-between mb-2' }, [
                                    React.createElement('div', { key: 'line-id', className: `font-bold ${lineError.coords ? 'text-red-800' : 'text-gray-800'}` }, 
                                        `Line: ${lineError.line_id}`
                                    ),
                                    React.createElement('div', { key: 'error-count', className: `text-sm px-2 py-1 rounded ${lineError.coords ? 'bg-red-200 text-red-800' : 'bg-gray-200 text-gray-800'}` }, 
                                        analysisMethod === 'chunks' ? 
                                        `${lineError.chunk_count || lineError.error_count} words` :
                                        `${lineError.error_count} errors`
                                    )
                                ]),
                                React.createElement('div', { key: 'line-text', className: `text-sm ${lineError.coords ? 'text-red-700' : 'text-gray-700'} mb-2 font-mono` }, 
                                    lineError.text ? (lineError.text.length > 60 ? lineError.text.substring(0, 60) + '...' : lineError.text) : 'No text'
                                ),
                                React.createElement('div', { key: 'details', className: `text-xs ${lineError.coords ? 'text-red-600' : 'text-gray-600'}` }, [

                                    analysisMethod === 'likelihood' && lineError.error_words && lineError.error_words.length > 0 && 
                                        ` • Error words: ${lineError.error_words.slice(0, 3).join(', ')}${lineError.error_words.length > 3 ? '...' : ''}`,
                                    
                                    analysisMethod === 'chunks' && lineError.error_chunks && lineError.error_chunks.length > 0 && 
                                        ` • Words: ${lineError.error_chunks.slice(0, 2).join(', ')}${lineError.error_chunks.length > 2 ? '...' : ''}`
                                ].filter(Boolean).join(''))
                            ])
                        )
                    ]) :
                    React.createElement('div', { key: 'no-errors', className: 'text-center py-8 text-green-600' }, [
                        React.createElement(CheckCircle, { key: 'icon', className: 'w-12 h-12 mx-auto mb-2' }),
                        React.createElement('div', { key: 'title', className: 'font-medium' }, 
                            analysisMethod === 'chunks' ? 'No chunk errors predicted' : 'No word errors predicted'
                        ),
                        React.createElement('div', { key: 'subtitle', className: 'text-sm' }, 
                            analysisMethod === 'chunks' ? 'All text chunks predicted as good quality' : 'All words predicted as good OCR quality'
                        )
                    ])
            ])
        ]);
    };

    return React.createElement('div', { 
        className: `right-panel ${isCollapsed ? 'collapsed' : ''}`,
        style: { position: 'relative' }
    }, [
        // Toggle button
        React.createElement('button', {
            key: 'toggle',
            className: 'right-panel-toggle',
            onClick: onToggleCollapse,
            title: isCollapsed ? 'Show analysis panel' : 'Hide analysis panel'
        }, React.createElement(isCollapsed ? ChevronLeft : ChevronRight)),

        // When collapsed, show a minimal score
        isCollapsed && React.createElement('div', {
            key: 'collapsed-indicator',
            className: 'collapsed-content',
            onClick: onToggleCollapse
        }, [
            React.createElement('div', { 
                key: 'indicator-text',
                className: 'collapsed-text' 
            }, 'Analysis'),
            React.createElement('div', { 
                key: 'score-indicator',
                className: 'collapsed-score' 
            }, currentPageData ? (currentPageData.stats.overallScore * 100).toFixed(0) + '%' : '--')
        ]),

        // When expanded, full content
        !isCollapsed && React.createElement('div', {
            key: 'panel-content',
            className: 'right-panel-content'
        }, analysisMethod === 'supervised' ? renderSupervisedRightPanel() : renderRegularRightPanel())
    ]);
};

// Helper function for supervised right panel
const renderSupervisedRightPanel = (currentPageData, displayErrors, getScoreIcon, getScoreColor, getReadabilityCategory) => {
    return React.createElement('div', { key: 'right-panel', className: 'space-y-6 overflow-y-auto h-full' }, [
        // Overall Quality Score 
        React.createElement('div', { key: 'quality-score', className: 'bg-white rounded-lg shadow-lg p-6' }, [
            React.createElement('h3', { key: 'score-title', className: 'text-lg font-semibold text-gray-800 mb-4 flex items-center' }, [
                getScoreIcon(currentPageData.stats.overallScore),
                React.createElement('span', { key: 'text', className: 'ml-2' }, 'Overall Quality Score')
            ]),
            React.createElement('div', { key: 'score-content', className: 'text-center' }, [
                React.createElement('div', { 
                    key: 'percentage',
                    className: `text-4xl font-bold ${getScoreColor(currentPageData.stats.overallScore)}`
                }, (currentPageData.stats.overallScore * 100).toFixed(1) + '%'),
                React.createElement('div', { 
                    key: 'readability',
                    className: `inline-flex items-center px-3 py-1 rounded-full text-sm font-medium mt-2 ${getReadabilityCategory(currentPageData.stats.overallScore).color} ${getReadabilityCategory(currentPageData.stats.overallScore).bgColor}`
                }, getReadabilityCategory(currentPageData.stats.overallScore).text),
                React.createElement('div', { key: 'progress-container', className: 'w-full bg-gray-200 rounded-full h-3 mt-3' },
                    React.createElement('div', {
                        className: `h-3 rounded-full ${
                            currentPageData.stats.overallScore >= 0.9 ? 'bg-green-600' :
                            currentPageData.stats.overallScore >= 0.7 ? 'bg-yellow-600' : 'bg-red-600'
                        }`,
                        style: createProgressStyle(currentPageData.stats.overallScore * 100)
                    })
                )
            ])
        ]),

        // Supervised
        React.createElement('div', { key: 'metrics', className: 'bg-white rounded-lg shadow-lg p-6' }, [
            React.createElement('h3', { key: 'metrics-title', className: 'text-lg font-semibold text-gray-800 mb-4' }, 
                'Supervised Analysis Metrics'
            ),
            React.createElement('div', { key: 'metrics-grid', className: 'grid grid-cols-2 gap-4' }, [
                React.createElement('div', { key: 'words', className: 'text-center p-3 bg-green-50 rounded-lg' }, [
                    React.createElement('div', { key: 'value', className: 'text-2xl font-bold text-green-600' },
                        currentPageData.stats.wordCount
                    ),
                    React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 'Total Words')
                ]),
                React.createElement('div', { key: 'model-type', className: 'text-center p-3 bg-purple-50 rounded-lg' }, [
                   React.createElement('div', { key: 'value', className: 'text-lg font-bold text-purple-600' },
                        currentPageData.stats.pageErrorScore ? 
                            parseInt(currentPageData.stats.pageErrorScore * currentPageData.stats.wordCount)
                            : 'N/A'
                    ),
                    React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 'Predicted Word Errors')
                ]),
                React.createElement('div', { key: 'total-lines', className: 'text-center p-3 bg-blue-50 rounded-lg' }, [
                    React.createElement('div', { key: 'value', className: 'text-2xl font-bold text-blue-600' },
                        displayErrors.length
                    ),
                    React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 'Total Lines')
                ]),
                React.createElement('div', { key: 'error-score', className: 'text-center p-3 bg-red-50 rounded-lg' }, [
                    React.createElement('div', { key: 'value', className: 'text-2xl font-bold text-red-600' },
                        currentPageData.stats.pageErrorScore ? (currentPageData.stats.pageErrorScore * 100).toFixed(1) + '%' : 'N/A'
                    ),
                    React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 'Page Error Score')
                ])
            ])
        ]),

        // Supervised method
        React.createElement('div', { key: 'method-info', className: 'bg-white rounded-lg shadow-lg p-6' }, [
            React.createElement('h3', { key: 'info-title', className: 'text-lg font-semibold text-gray-800 mb-4' }, 
                'Supervised Learning Approach'
            ),
            React.createElement('div', { key: 'info-content', className: 'space-y-4' }, [
                React.createElement('div', { key: 'description', className: 'p-4 bg-purple-50 rounded-lg' }, [
                    React.createElement('h4', { key: 'desc-title', className: 'font-semibold text-purple-800 mb-2' }, 'Model Architecture'),
                    React.createElement('p', { key: 'desc-text', className: 'text-sm text-purple-700' }, 
                        'This approach uses a regression model trained on page embeddings and structural features to predict page-level error score.'
                    )
                ]),
                currentPageData.stats.pageErrorScore && React.createElement('div', { key: 'score-details', className: 'p-4 bg-gray-50 rounded-lg' }, [
                    React.createElement('h4', { key: 'score-title', className: 'font-semibold text-gray-800 mb-2' }, 'Prediction Details'),
                    React.createElement('div', { key: 'score-items', className: 'space-y-2 text-sm' }, [
                        React.createElement('div', { key: 'raw-score' }, [
                            React.createElement('span', { className: 'font-medium' }, 'Raw Error Score: '),
                            React.createElement('span', { className: 'font-mono text-red-600' }, 
                                currentPageData.stats.pageErrorScore.toFixed(6)
                            )
                        ]),
                        React.createElement('div', { key: 'quality-score' }, [
                            React.createElement('span', { className: 'font-medium' }, 'Quality Score: '),
                            React.createElement('span', { className: 'font-mono text-green-600' }, 
                                (currentPageData.stats.overallScore * 100).toFixed(1) + '%'
                            )
                        ]),
                        React.createElement('div', { key: 'prediction-type' }, [
                            React.createElement('span', { className: 'font-medium' }, 'Prediction Type: '),
                            React.createElement('span', { className: 'text-purple-600' }, 'Page-level error rate')
                        ])
                    ])
                ])
            ])
        ])
    ]);
};

// Helper function for right panel for likelihood and chunks
const renderRegularRightPanel = (currentPageData, displayErrors, predictionErrors, analysisMethod, availableMethods, getScoreIcon, getScoreColor, getReadabilityCategory) => {
    return React.createElement('div', { key: 'right-panel', className: 'space-y-6 overflow-y-auto h-full' }, [
        // Overall Quality Score
        React.createElement('div', { key: 'quality-score', className: 'bg-white rounded-lg shadow-lg p-6' }, [
            React.createElement('h3', { key: 'score-title', className: 'text-lg font-semibold text-gray-800 mb-4 flex items-center' }, [
                getScoreIcon(currentPageData.stats.overallScore),
                React.createElement('span', { key: 'text', className: 'ml-2' }, 'Overall Quality Score')
            ]),
            React.createElement('div', { key: 'score-content', className: 'text-center' }, [
                React.createElement('div', { 
                    key: 'percentage',
                    className: `text-4xl font-bold ${getScoreColor(currentPageData.stats.overallScore)}`
                }, (currentPageData.stats.overallScore * 100).toFixed(1) + '%'),
                React.createElement('div', { 
                    key: 'readability',
                    className: `inline-flex items-center px-3 py-1 rounded-full text-sm font-medium mt-2 ${getReadabilityCategory(currentPageData.stats.overallScore).color} ${getReadabilityCategory(currentPageData.stats.overallScore).bgColor}`
                }, getReadabilityCategory(currentPageData.stats.overallScore).text),
                React.createElement('div', { key: 'progress-container', className: 'w-full bg-gray-200 rounded-full h-3 mt-3' },
                    React.createElement('div', {
                        className: `h-3 rounded-full ${
                            currentPageData.stats.overallScore >= 0.9 ? 'bg-green-600' :
                            currentPageData.stats.overallScore >= 0.7 ? 'bg-yellow-600' : 'bg-red-600'
                        }`,
                        style: createProgressStyle(currentPageData.stats.overallScore * 100)
                    })
                )
            ])
        ]),

        // Key Metrics
        React.createElement('div', { key: 'metrics', className: 'bg-white rounded-lg shadow-lg p-6' }, [
            React.createElement('h3', { key: 'metrics-title', className: 'text-lg font-semibold text-gray-800 mb-4' }, 
                `${availableMethods.find(m => m.id === analysisMethod)?.name || 'Analysis'} Metrics`
            ),
            React.createElement('div', { key: 'metrics-grid', className: 'grid grid-cols-2 gap-4' }, [
                React.createElement('div', { key: 'words', className: 'text-center p-3 bg-green-50 rounded-lg' }, [
                    React.createElement('div', { key: 'value', className: 'text-2xl font-bold text-green-600' },
                        currentPageData.stats.wordCount
                    ),
                    React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 'Total Words')
                ]),
                React.createElement('div', { key: 'errors', className: 'text-center p-3 bg-orange-50 rounded-lg' }, [
                    React.createElement('div', { key: 'value', className: 'text-2xl font-bold text-orange-600' },
                        currentPageData.stats.errorCount
                    ),
                    React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 
                        analysisMethod === 'chunks' ? 'Predicted Words Errors' : 'Predicted Word Errors'
                    )
                ]),
                React.createElement('div', { key: 'total-lines', className: 'text-center p-3 bg-blue-50 rounded-lg' }, [
                    React.createElement('div', { key: 'value', className: 'text-2xl font-bold text-blue-600' },
                        displayErrors.length
                    ),
                    React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 'Total Lines (Display)')
                ]),
                React.createElement('div', { key: 'error-lines', className: 'text-center p-3 bg-red-50 rounded-lg' }, [
                    React.createElement('div', { key: 'value', className: 'text-2xl font-bold text-red-600' },
                        predictionErrors.length
                    ),
                    React.createElement('div', { key: 'label', className: 'text-sm text-gray-600' }, 'Lines with Predictions')
                ])
            ])
        ]),

        // Error Details
        React.createElement('div', { key: 'prediction-errors', className: 'bg-white rounded-lg shadow-lg p-6' }, [
            React.createElement('h3', { key: 'errors-title', className: 'text-lg font-semibold text-gray-800 mb-4' }, 
                `${analysisMethod === 'chunks' ? 'Chunk' : 'Word'} Error Predictions`
            ),
            predictionErrors.length > 0 ? 
                React.createElement('div', { key: 'errors-list', className: 'space-y-3 max-h-64 overflow-y-auto' }, [
                    ...predictionErrors.slice(0, predictionErrors.length).map((lineError, index) => 
                        React.createElement('div', { 
                            key: index, 
                            className: `p-4 rounded-lg ${
                                lineError.coords ? 'bg-red-50 border border-red-200' : 'bg-gray-50'
                            }` 
                        }, [
                            React.createElement('div', { key: 'line-header', className: 'flex items-center justify-between mb-2' }, [
                                React.createElement('div', { key: 'line-id', className: `font-bold ${lineError.coords ? 'text-red-800' : 'text-gray-800'}` }, 
                                    `Line: ${lineError.line_id}`
                                ),
                                React.createElement('div', { key: 'error-count', className: `text-sm px-2 py-1 rounded ${lineError.coords ? 'bg-red-200 text-red-800' : 'bg-gray-200 text-gray-800'}` }, 
                                    analysisMethod === 'chunks' ? 
                                    `${lineError.chunk_count || lineError.error_count} words` :
                                    `${lineError.error_count} errors`
                                )
                            ]),
                            React.createElement('div', { key: 'line-text', className: `text-sm ${lineError.coords ? 'text-red-700' : 'text-gray-700'} mb-2 font-mono` }, 
                                lineError.text ? (lineError.text.length > 60 ? lineError.text.substring(0, 60) + '...' : lineError.text) : 'No text'
                            ),
                            React.createElement('div', { key: 'details', className: `text-xs ${lineError.coords ? 'text-red-600' : 'text-gray-600'}` }, [
                                analysisMethod === 'likelihood' && lineError.error_words && lineError.error_words.length > 0 && 
                                    ` • Error words: ${lineError.error_words.slice(0, 3).join(', ')}${lineError.error_words.length > 3 ? '...' : ''}`,
                                
                                analysisMethod === 'chunks' && lineError.error_chunks && lineError.error_chunks.length > 0 && 
                                    ` • Words: ${lineError.error_chunks.slice(0, 2).join(', ')}${lineError.error_chunks.length > 2 ? '...' : ''}`
                            ].filter(Boolean).join(''))
                        ])
                    )
                ]) :
                React.createElement('div', { key: 'no-errors', className: 'text-center py-8 text-green-600' }, [
                    React.createElement(CheckCircle, { key: 'icon', className: 'w-12 h-12 mx-auto mb-2' }),
                    React.createElement('div', { key: 'title', className: 'font-medium' }, 
                        analysisMethod === 'chunks' ? 'No chunk errors predicted' : 'No word errors predicted'
                    ),
                    React.createElement('div', { key: 'subtitle', className: 'text-sm' }, 
                        analysisMethod === 'chunks' ? 'All text chunks predicted as good quality' : 'All words predicted as good OCR quality'
                    )
                ])
        ])
    ]);
};