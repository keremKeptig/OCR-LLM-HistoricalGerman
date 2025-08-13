// React hooks available globally for all components
const { useState, useEffect, useRef, useCallback, useMemo } = React;
window.useState = useState;
window.useEffect = useEffect;
window.useRef = useRef;
window.useCallback = useCallback;
window.useMemo = useMemo;

const createProgressStyle = (percentage) => ({ width: percentage + '%' });

const createTransformStyle = (scale, x, y) => ({
    transform: `scale(${scale}) translate(${x / scale}px, ${y / scale}px)`,
    transformOrigin: '0 0'
});
window.createProgressStyle = createProgressStyle;
window.createTransformStyle = createTransformStyle;