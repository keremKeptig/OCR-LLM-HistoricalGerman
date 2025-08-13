// Pages Sidebar Component
const PagesSidebar = ({ pages, currentPage, onPageSelect, analysisMethod, sortOrder, onSortChange, isCollapsed, onToggleCollapse }) => {
    const [searchQuery, setSearchQuery] = useState('');

    // Get score for sorting
    const getScoreForSort = (p) => 
        (typeof p.quality === 'number' ? p.quality : p.stats?.overallScore) ?? 0;

    // Filter and sort pages based on search and sort order
    const filteredPages = useMemo(() => {
        let filtered = pages.filter(page => 
            page.filename.toLowerCase().includes(searchQuery.toLowerCase())
        );

        // Apply sorting
        switch (sortOrder) {
            case 'best-to-worst':
                return [...filtered].sort((a, b) => getScoreForSort(b) - getScoreForSort(a));
            case 'worst-to-best':
                return [...filtered].sort((a, b) => getScoreForSort(a) - getScoreForSort(b));
            case 'original':
            default:
                return filtered;
        }
    }, [pages, searchQuery, sortOrder]);

    const getQualityClass = (score) => {
        if (score >= 0.9) return 'quality-good';
        if (score >= 0.7) return 'quality-fair';
        if (score >= 0.35) return 'quality-poor';
        return 'quality-very-problematic';
    };

    const getQualityText = (score) => {
        if (score >= 0.9) return 'Good';
        if (score >= 0.7) return 'Fair';
        if (score >= 0.35) return 'Poor';
        return 'Very Problematic';
    };

    const getStatusDotColor = (score) => {
        if (score >= 0.9) return '#10b981';
        if (score >= 0.8) return '#f59e0b';
        if (score >= 0.7) return '#f97316';
        return '#ef4444';
    };

    return React.createElement('div', { 
        className: `pages-sidebar ${isCollapsed ? 'collapsed' : ''}`,
        style: { position: 'relative' },
        onClick: isCollapsed ? onToggleCollapse : undefined
    }, [
        // Toggle button
        React.createElement('button', {
            key: 'toggle',
            className: 'sidebar-toggle',
            onClick: (e) => {
                e.stopPropagation();
                onToggleCollapse();
            },
            title: isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'
        }, React.createElement(isCollapsed ? Menu : X)),

        // When collapsed, show a better visual indicator
        isCollapsed && React.createElement('div', {
            key: 'collapsed-indicator',
            className: 'collapsed-indicator',
            title: 'Click to expand pages sidebar'
        }, [
            React.createElement('div', {
                key: 'icon-container',
                className: 'collapsed-icon'
            }, React.createElement(FileText, { key: 'icon' })),
            
            React.createElement('div', {
                key: 'count',
                className: 'collapsed-count'
            }, pages.length),
            
            React.createElement('div', {
                key: 'label',
                className: 'collapsed-label'
            }, 'PAGES'),
            
            // current page indicator with page number
            pages.length > 0 && React.createElement('div', {
                key: 'current-page-info',
                style: {
                    marginTop: '12px',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '4px'
                }
            }, [
                React.createElement('div', {
                    key: 'page-number',
                    style: {
                        fontSize: '9px',
                        color: '#6b7280',
                        fontWeight: '600'
                    }
                }, `${currentPage + 1}/${pages.length}`),
                React.createElement('div', {
                    key: 'progress-bar',
                    style: {
                        width: '24px',
                        height: '3px',
                        background: '#e5e7eb',
                        borderRadius: '2px',
                        overflow: 'hidden'
                    }
                }, React.createElement('div', {
                    style: {
                        width: `${((currentPage + 1) / pages.length) * 100}%`,
                        height: '100%',
                        background: 'linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%)',
                        transition: 'width 0.3s ease'
                    }
                }))
            ])
        ]),

        // Header for when expanded
        !isCollapsed && React.createElement('div', { key: 'header', className: 'pages-header' }, [
            React.createElement('div', { key: 'title', className: 'pages-title mb-3' }, 'Document Pages'),
            React.createElement('input', {
                key: 'search',
                type: 'text',
                placeholder: 'Search pages...',
                value: searchQuery,
                onChange: (e) => setSearchQuery(e.target.value),
                className: 'pages-search mb-3'
            }),
            React.createElement('select', {
                key: 'sort-select',
                value: sortOrder,
                onChange: (e) => onSortChange(e.target.value),
                className: 'w-full px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-medium',
                title: 'Sort pages by score'
            }, [
                React.createElement('option', { key: 'original', value: 'original' }, 'Sort by: Original Order'),
                React.createElement('option', { key: 'best-to-worst', value: 'best-to-worst' }, 'Sort by: Best to Worst'),
                React.createElement('option', { key: 'worst-to-best', value: 'worst-to-best' }, 'Sort by: Worst to Best')
            ])
        ]),

        // Pages count, only show when expanded
        !isCollapsed && React.createElement('div', { key: 'count', className: 'pages-count' }, 
            `${filteredPages.length} of ${pages.length} pages${searchQuery ? ` matching "${searchQuery}"` : ''}`
        ),

        // Pages list, only show when expanded
        !isCollapsed && React.createElement('div', { key: 'list', className: 'pages-list' }, 
            filteredPages.map((page, index) => 
                React.createElement('div', {
                    key: page.id,
                    className: `page-item ${pages[currentPage]?.id === page.id ? 'selected' : ''}`,
                    onClick: () => {
                        const originalIndex = pages.findIndex(p => p.id === page.id);
                        onPageSelect(originalIndex);
                    }
                }, [
                    React.createElement('div', { 
                        key: 'thumbnail', 
                        className: 'page-thumbnail',
                        title: page.filename
                    }, [
                        React.createElement('img', {
                            key: 'img',
                            src: page.imageUrl,
                            alt: page.filename,
                            onError: (e) => {
                                e.target.style.display = 'none';
                                e.target.nextSibling.style.display = 'block';
                            }
                        }),
                        React.createElement('div', {
                            key: 'fallback',
                            style: { display: 'none' }
                        }, page.id)
                    ]),
                    React.createElement('div', { key: 'info', className: 'page-info' }, [
                        React.createElement('div', { 
                            key: 'name', 
                            className: 'page-name',
                            title: page.filename
                        }, page.filename.length > 20 ? `${page.filename.substring(0, 20)}...` : page.filename),
                        React.createElement('div', { key: 'details', className: 'page-details' }, 
                            `${(page.quality * 100).toFixed(0)}% `
                        ),
                        
                        React.createElement('div', { 
                            key: 'quality', 
                            className: `page-quality ${getQualityClass(page.quality)}`
                        }, getQualityText(page.quality))
                    ]),
                    React.createElement('div', { key: 'status', className: 'page-status' }, [
                        React.createElement('div', {
                            key: 'dot',
                            className: 'status-dot',
                            style: { backgroundColor: getStatusDotColor(page.quality) }
                        })
                    ])
                ])
            )
        )
    ]);
};