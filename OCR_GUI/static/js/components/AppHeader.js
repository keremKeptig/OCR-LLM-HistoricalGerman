// first line stays the same:
const AppHeader = ({ onRunEvaluation, viewType, onViewTypeChange, hasOcrText, hasTextRegionsGT, hasErrorOverlayPred, analysisMethod, onOpenBooks }) => {
    return React.createElement('header', { className: 'app-header' }, 
        React.createElement('div', { className: 'header-content' }, [
            React.createElement('div', { key: 'title', className: 'header-title' },
                React.createElement('h1', {}, 'OCR Quality Estimation')
            ),
            React.createElement('nav', { key: 'nav', className: 'header-nav' }, [
                React.createElement(ViewTypeSelector, {
                    key: 'view-selector',
                    viewType: viewType,
                    onViewTypeChange: onViewTypeChange,
                    hasOcrText: hasOcrText,
                    hasTextRegionsGT: hasTextRegionsGT,
                    hasErrorOverlayPred: hasErrorOverlayPred,
                    analysisMethod: analysisMethod
                }),
                
                React.createElement('button', {
                    key: 'evaluation',
                    className: 'data-visualization-button',
                    onClick: onRunEvaluation
                }, [
                    React.createElement(BarChart, { key: 'icon' }),
                    React.createElement('span', { key: 'text' }, 'Evaluation')
                ]),

                React.createElement('button', {
                    key: 'books',
                    className: 'data-visualization-button',
                    onClick: onOpenBooks
                }, [
                    React.createElement(BarChart, { key: 'icon' }),
                    React.createElement('span', { key: 'text' }, 'Average Books Score')
                ])
            ])
        ])
    );
};
