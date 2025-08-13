// Sort Dropdown Component  
const SortDropdown = ({ sortOrder, onSortChange }) => {
    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef(null);
    
    const sortOptions = [
        { id: 'original', name: 'Original Order', icon: 'M3 4h13M3 8h9m-9 4h9m5-4v12l-4-4m4-8v12' },
        { id: 'best-to-worst', name: 'Best to Worst', icon: 'M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12' },
        { id: 'worst-to-best', name: 'Worst to Best', icon: 'M3 4h13M3 8h9m-9 4h6m4 0l4 4m0 0l4-4m-4 4V8' }
    ];
    
    const selectedSort = sortOptions.find(opt => opt.id === sortOrder) || sortOptions[0];

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setIsOpen(false);
            }
        };

        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside);
            return () => document.removeEventListener('mousedown', handleClickOutside);
        }
    }, [isOpen]);
};

// Dropdown Component
const MethodDropdown = ({ selectedMethod, onMethodChange, availableMethods }) => {
    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef(null);
    const selectedMethodInfo = availableMethods.find(m => m.id === selectedMethod);

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setIsOpen(false);
            }
        };

        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside);
            return () => document.removeEventListener('mousedown', handleClickOutside);
        }
    }, [isOpen]);

    return React.createElement('div', { 
        ref: dropdownRef,
        className: 'relative' 
    }, [
        React.createElement('button', {
            key: 'dropdown-button',
            onClick: () => setIsOpen(!isOpen),
            className: 'px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors flex items-center gap-2 min-w-48',
            title: 'Select analysis method'
        }, [
            React.createElement(Settings, { key: 'icon' }),
            React.createElement('span', { key: 'text', className: 'text-sm font-medium' }, 
                selectedMethodInfo ? selectedMethodInfo.name : 'Select Method'
            ),
            React.createElement('svg', {
                key: 'chevron',
                width: "12", height: "12", viewBox: "0 0 24 24", fill: "none", 
                stroke: "currentColor", strokeWidth: "2",
                className: `transition-transform ${isOpen ? 'rotate-180' : ''}`
            }, React.createElement('path', { d: "m6 9 6 6 6-6" }))
        ]),
        
        isOpen && React.createElement('div', {
            key: 'dropdown-menu',
            className: 'absolute top-full left-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-50 min-w-80',
            style: { maxHeight: '300px', overflowY: 'auto' }
        }, [
            React.createElement('div', {
                key: 'dropdown-header',
                className: 'px-3 py-2 border-b border-gray-100 bg-gray-50 rounded-t-lg'
            }, React.createElement('div', {
                className: 'text-sm font-semibold text-gray-700'
            }, 'Error Analysis Method')),
            
            React.createElement('div', { key: 'dropdown-options', className: 'py-1' },
                availableMethods.map(method => 
                    React.createElement('button', {
                        key: method.id,
                        onClick: () => {
                            if (method.available) {
                                onMethodChange(method.id);
                                setIsOpen(false);
                            }
                        },
                        disabled: !method.available,
                        className: `w-full text-left px-3 py-3 hover:bg-gray-50 transition-colors border-b border-gray-50 last:border-b-0 ${
                            selectedMethod === method.id ? 'bg-blue-50 text-blue-700' : ''
                        } ${!method.available ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`
                    }, [
                        React.createElement('div', { key: 'method-name', className: 'font-medium text-sm' }, [
                            method.name,
                            selectedMethod === method.id && React.createElement('span', { 
                                key: 'selected-indicator', 
                                className: 'ml-2 text-blue-600' 
                            }, '✓')
                        ]),
                        React.createElement('div', { key: 'method-desc', className: 'text-xs text-gray-600 mt-1' }, 
                            method.description
                        ),
                        !method.available && React.createElement('div', { 
                            key: 'status', 
                            className: 'text-xs text-orange-600 mt-1 font-medium' 
                        }, 'Not Available')
                    ])
                )
            )
        ])
    ]);
};