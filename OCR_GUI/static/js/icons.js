const ChevronLeft = () => React.createElement('svg', {
    width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, React.createElement('path', { d: "m15 18-6-6 6-6" }));

const ChevronRight = () => React.createElement('svg', {
    width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, React.createElement('path', { d: "m9 18 6-6-6-6" }));

const Menu = () => React.createElement('svg', {
    width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('line', { key: 1, x1: "3", y1: "6", x2: "21", y2: "6" }),
    React.createElement('line', { key: 2, x1: "3", y1: "12", x2: "21", y2: "12" }),
    React.createElement('line', { key: 3, x1: "3", y1: "18", x2: "21", y2: "18" })
]);

const X = () => React.createElement('svg', {
    width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('path', { key: 1, d: "m18 6-12 12" }),
    React.createElement('path', { key: 2, d: "m6 6 12 12" })
]);

const FileText = () => React.createElement('svg', {
    width: "20", height: "20", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('path', { key: 1, d: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" }),
    React.createElement('polyline', { key: 2, points: "14 2 14 8 20 8" }),
    React.createElement('line', { key: 3, x1: "16", y1: "13", x2: "8", y2: "13" }),
    React.createElement('line', { key: 4, x1: "16", y1: "17", x2: "8", y2: "17" }),
    React.createElement('polyline', { key: 5, points: "10 9 9 9 8 9" })
]);

const FileImage = () => React.createElement('svg', {
    width: "20", height: "20", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('path', { key: 1, d: "M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z" }),
    React.createElement('polyline', { key: 2, points: "14 2 14 8 20 8" }),
    React.createElement('circle', { key: 3, cx: "10", cy: "13", r: "2" }),
    React.createElement('path', { key: 4, d: "M20 17l-5.5-5.5L9 17H20z" })
]);

const Layers = () => React.createElement('svg', {
    width: "20", height: "20", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('polygon', { key: 1, points: "12 2 2 7 12 12 22 7 12 2" }),
    React.createElement('polyline', { key: 2, points: "2 17 12 22 22 17" }),
    React.createElement('polyline', { key: 3, points: "2 12 12 17 22 12" })
]);

const Target = () => React.createElement('svg', {
    width: "20", height: "20", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('circle', { key: 1, cx: "12", cy: "12", r: "10" }),
    React.createElement('circle', { key: 2, cx: "12", cy: "12", r: "6" }),
    React.createElement('circle', { key: 3, cx: "12", cy: "12", r: "2" })
]);

const RefreshCw = () => React.createElement('svg', {
    width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('polyline', { key: 1, points: "23 4 23 10 17 10" }),
    React.createElement('polyline', { key: 2, points: "1 20 1 14 7 14" }),
    React.createElement('path', { key: 3, d: "M20.49 15a9 9 0 1 1-2.12-9.36L23 10" }),
    React.createElement('path', { key: 4, d: "M3.51 9a9 9 0 0 1 2.12 9.36L1 14" })
]);

const Settings = () => React.createElement('svg', {
    width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('circle', { key: 1, cx: "12", cy: "12", r: "3" }),
    React.createElement('path', { key: 2, d: "m12 1 1.9 5.7a8.2 8.2 0 0 1 1.6.8l5.7-1.9 1.4 2.4-5.7 1.9a8.2 8.2 0 0 1 0 1.6l5.7 1.9-1.4 2.4-5.7-1.9a8.2 8.2 0 0 1-1.6.8L12 23l-1.9-5.7a8.2 8.2 0 0 1-1.6-.8l-5.7 1.9-1.4-2.4 5.7-1.9a8.2 8.2 0 0 1 0-1.6L1.4 11.5 2.8 9.1l5.7 1.9a8.2 8.2 0 0 1 1.6-.8L12 1z" })
]);

const ZoomIn = () => React.createElement('svg', {
    width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('circle', { key: 1, cx: "11", cy: "11", r: "8" }),
    React.createElement('path', { key: 2, d: "M21 21l-4.35-4.35" }),
    React.createElement('line', { key: 3, x1: "11", y1: "8", x2: "11", y2: "14" }),
    React.createElement('line', { key: 4, x1: "8", y1: "11", x2: "14", y2: "11" })
]);

const ZoomOut = () => React.createElement('svg', {
    width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('circle', { key: 1, cx: "11", cy: "11", r: "8" }),
    React.createElement('path', { key: 2, d: "M21 21l-4.35-4.35" }),
    React.createElement('line', { key: 3, x1: "8", y1: "11", x2: "14", y2: "11" })
]);

const RotateCw = () => React.createElement('svg', {
    width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('polyline', { key: 1, points: "23 4 23 10 17 10" }),
    React.createElement('path', { key: 2, d: "M20.49 15a9 9 0 1 1-2.12-9.36L23 10" })
]);

const CheckCircle = () => React.createElement('svg', {
    width: "20", height: "20", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('path', { key: 1, d: "M22 11.08V12a10 10 0 1 1-5.93-9.14" }),
    React.createElement('polyline', { key: 2, points: "22 4 12 14.01 9 11.01" })
]);

const AlertTriangle = () => React.createElement('svg', {
    width: "20", height: "20", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('path', { key: 1, d: "m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" }),
    React.createElement('line', { key: 2, x1: "12", y1: "9", x2: "12", y2: "13" }),
    React.createElement('line', { key: 3, x1: "12", y1: "17", x2: "12.01", y2: "17" })
]);

const Eye = () => React.createElement('svg', {
    width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('path', { key: 1, d: "M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" }),
    React.createElement('circle', { key: 2, cx: "12", cy: "12", r: "3" })
]);

const EyeOff = () => React.createElement('svg', {
    width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('path', { key: 1, d: "M9.88 9.88a3 3 0 1 0 4.24 4.24" }),
    React.createElement('path', { key: 2, d: "M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 11 8 11 8a13.16 13.16 0 0 1-1.67 2.68" }),
    React.createElement('path', { key: 3, d: "M6.61 6.61A13.526 13.526 0 0 0 1 12s4 8 11 8a9.74 9.74 0 0 0 5.39-1.61" }),
    React.createElement('line', { key: 4, x1: "2", y1: "2", x2: "22", y2: "22" })
]);

const Brain = () => React.createElement('svg', {
    width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('path', { key: 1, d: "M12 2a3 3 0 0 0-3 3 4 4 0 0 0-6 1c-1 5 4 10 12 10s13-5 12-10a4 4 0 0 0-6-1 3 3 0 0 0-3-3Z" }),
    React.createElement('path', { key: 2, d: "M12 22V2" })
]);

const SearchIcon = () => React.createElement('svg', {
    width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('circle', { key: 1, cx: "11", cy: "11", r: "8" }),
    React.createElement('path', { key: 2, d: "m21 21-4.35-4.35" })
]);

const BarChart = () => React.createElement('svg', {
    width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", 
    stroke: "currentColor", strokeWidth: "2"
}, [
    React.createElement('line', { key: 1, x1: "12", y1: "20", x2: "12", y2: "10" }),
    React.createElement('line', { key: 2, x1: "18", y1: "20", x2: "18", y2: "4" }),
    React.createElement('line', { key: 3, x1: "6", y1: "20", x2: "6", y2: "16" })
]);

swindow.ChevronLeft = ChevronLeft;
window.ChevronRight = ChevronRight;
window.Menu = Menu;
window.X = X;
window.FileText = FileText;
window.FileImage = FileImage;
window.Layers = Layers;
window.Target = Target;
window.RefreshCw = RefreshCw;
window.Settings = Settings;
window.ZoomIn = ZoomIn;
window.ZoomOut = ZoomOut;
window.RotateCw = RotateCw;
window.CheckCircle = CheckCircle;
window.AlertTriangle = AlertTriangle;
window.Eye = Eye;
window.EyeOff = EyeOff;
window.Brain = Brain;
window.SearchIcon = SearchIcon;
window.BarChart = BarChart;