// intsharp docs - Client-side search

const searchIndex = [
    { title: 'Home', url: 'index.html', keywords: 'home overview introduction features quick start' },
    { title: 'Installation', url: 'install.html', keywords: 'install setup pip requirements dependencies python' },
    { title: 'Running Simulations', url: 'guide.html', keywords: 'run simulation yaml config example tutorial guide per-field sharpening contour compare' },
    { title: 'Components', url: 'components.html', keywords: 'components solver upwind timestepper euler rk4 sharpening pm cl boundary periodic neumann dirichlet monitor console png pdf svg gif mp4 hdf5 txt curve compare compare_fields dg p1 p2 p3 hllc hlle ausm_plus_up convergence riemann' },
    { title: 'Unit Tests', url: 'unit-tests.html', keywords: 'unit test validation tanh tanh_circle 2d sharpening gif txt curve output circle radial revolution compare per-field' },
];

const contentIndex = {
    'index.html': 'intsharp modular yaml simulation framework 1d 2d interface advection sharpening pydantic validation upwind euler rk4 pm cl periodic neumann dirichlet console png pdf gif hdf5 txt curve registry contour compare',
    'install.html': 'install installation clone git pip requirements numpy matplotlib scipy pydantic pyyaml tqdm h5py imageio python virtual environment venv',
    'guide.html': 'running simulation yaml config configuration domain time velocity fields initial condition image png initial_condition_image boundary solver timestepper sharpening output monitors tanh hat advection revolution unit test validation example 2d circle per-field sharpening gif mp4 svg compare compare_fields euler_spatial_discretization flux_calculator dg_order euler mode',
    'components.html': 'upwind advection solver euler rk4 runge kutta timestepper pm parameswaran mandal cl chiu lin sharpening periodic neumann dirichlet boundary condition console progress png pdf svg vector image gif mp4 video animation hdf5 data txt text curve output monitor gradient divergence cfl courant 1d 2d meshgrid pcolormesh contour compare compare_fields dg p1 p2 p3 fv hllc hlle ausm_plus_up exact riemann convergence',
    'unit-tests.html': 'unit test validation tanh compare tanh_hat 10_rev tanh_circle 2d circle radial revolution periodic domain cell-centered cfl gif compare compare_fields per-field 1d sod euler dg p1 p2 p3 hllc convergence'
};

document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search');
    const navLinks = document.querySelectorAll('.nav-list a');
    
    if (!searchInput) return;
    
    searchInput.addEventListener('input', function() {
        const query = this.value.toLowerCase().trim();
        
        if (query === '') {
            // Show all links
            navLinks.forEach(link => {
                link.classList.remove('search-hidden');
            });
            return;
        }
        
        // Filter navigation links
        navLinks.forEach(link => {
            const page = link.getAttribute('data-page');
            const title = link.textContent.toLowerCase();
            const url = link.getAttribute('href');
            
            // Check title match
            let matches = title.includes(query);
            
            // Check index match
            if (!matches) {
                const indexItem = searchIndex.find(item => item.url === url);
                if (indexItem && indexItem.keywords.includes(query)) {
                    matches = true;
                }
            }
            
            // Check content index match
            if (!matches && contentIndex[url]) {
                matches = contentIndex[url].includes(query);
            }
            
            if (matches) {
                link.classList.remove('search-hidden');
            } else {
                link.classList.add('search-hidden');
            }
        });
    });
    
    // Keyboard shortcut: Ctrl/Cmd + K to focus search
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            searchInput.focus();
            searchInput.select();
        }
    });
});
