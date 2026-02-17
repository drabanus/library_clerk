/**
 * Library Clerk - Interactive Knowledge Graph Visualization
 *
 * Supports three visualization modes:
 *   1. 2D Network Graph (vis.js) - traditional node-link diagram
 *   2. 3D Point Cloud (3d-force-graph) - spatial force-directed layout
 *   3. Timeline View - chronological research history
 *
 * Interactive features:
 *   - Click nodes to see details and neighborhood
 *   - Double-click to focus/expand
 *   - Hover for tooltips
 *   - Filter by node type and edge type
 *   - Search with highlighting
 *   - Zoom, pan, rotate (3D)
 *   - Cluster detection
 */

// ============================================================
// State
// ============================================================

let graphData = { nodes: [], edges: [] };
let currentView = '2d';
let visNetwork = null;
let forceGraph3D = null;
let activeNodeTypes = new Set();
let activeEdgeTypes = new Set();

const NODE_COLORS = {
    'Publication': '#58a6ff',
    'Author': '#3fb950',
    'Topic': '#d29922',
    'Keyword': '#8b949e',
    'Domain': '#f85149',
    'Methodology': '#bc8cff',
    'Concept': '#f778ba',
    'Entity': '#6e7681',
};

const NODE_SIZES = {
    'Publication': 18,
    'Author': 14,
    'Domain': 16,
    'Topic': 12,
    'Methodology': 12,
    'Concept': 11,
    'Keyword': 9,
    'Entity': 10,
};

// ============================================================
// Initialization
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    loadFullGraph();
    loadStats();

    document.getElementById('search-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') doSearch();
    });
});

async function loadFullGraph() {
    try {
        const resp = await fetch('/api/graph');
        graphData = await resp.json();
        populateFilters();
        renderCurrentView();
    } catch (err) {
        console.error('Failed to load graph:', err);
    }
}

async function loadStats() {
    try {
        const resp = await fetch('/api/stats');
        const stats = await resp.json();
        renderStats(stats);
    } catch (err) {
        console.error('Failed to load stats:', err);
    }
}

// ============================================================
// View Switching
// ============================================================

function switchView(view) {
    currentView = view;
    document.querySelectorAll('.graph-container').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('active'));

    document.querySelector(`[data-view="${view}"]`).classList.add('active');

    if (view === '2d') {
        document.getElementById('graph-2d').classList.add('active');
        render2DGraph();
    } else if (view === '3d') {
        document.getElementById('graph-3d').classList.add('active');
        render3DGraph();
    } else if (view === 'timeline') {
        document.getElementById('timeline-container').classList.add('active');
        renderTimeline();
    }
}

function renderCurrentView() {
    if (currentView === '2d') render2DGraph();
    else if (currentView === '3d') render3DGraph();
    else if (currentView === 'timeline') renderTimeline();
}

// ============================================================
// 2D Graph (vis.js)
// ============================================================

function render2DGraph() {
    const container = document.getElementById('graph-2d');
    const filtered = filterGraphData();

    const nodes = new vis.DataSet(filtered.nodes.map(n => ({
        id: n.id,
        label: truncateLabel(n.label || n.id, 25),
        title: buildTooltipHTML(n),
        color: {
            background: NODE_COLORS[n.node_type] || '#6e7681',
            border: NODE_COLORS[n.node_type] || '#6e7681',
            highlight: { background: '#fff', border: NODE_COLORS[n.node_type] || '#6e7681' },
            hover: { background: lighten(NODE_COLORS[n.node_type] || '#6e7681'), border: '#fff' },
        },
        size: NODE_SIZES[n.node_type] || 10,
        font: { color: '#e6edf3', size: 11, face: 'sans-serif' },
        shape: n.node_type === 'Publication' ? 'dot' : 'dot',
        borderWidth: n.node_type === 'Publication' ? 2 : 1,
        _raw: n,
    })));

    const edges = new vis.DataSet(filtered.edges.map((e, i) => ({
        id: `e_${i}`,
        from: e.source,
        to: e.target,
        label: '',
        title: `${e.relation || 'RELATED'} (${(e.weight || 0.5).toFixed(2)})`,
        color: { color: '#30363d', highlight: '#58a6ff', hover: '#484f58' },
        width: Math.max(1, (e.weight || 0.5) * 3),
        smooth: { type: 'continuous' },
        arrows: { to: { enabled: true, scaleFactor: 0.5 } },
    })));

    const options = {
        physics: {
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {
                gravitationalConstant: -80,
                centralGravity: 0.01,
                springLength: 120,
                springConstant: 0.08,
                damping: 0.4,
            },
            stabilization: { iterations: 150 },
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            zoomView: true,
            dragView: true,
            multiselect: true,
        },
        nodes: {
            borderWidth: 1,
            shadow: { enabled: true, size: 5, color: 'rgba(0,0,0,0.3)' },
        },
        edges: {
            smooth: { type: 'continuous' },
        },
    };

    if (visNetwork) visNetwork.destroy();
    visNetwork = new vis.Network(container, { nodes, edges }, options);

    // Click handler
    visNetwork.on('click', (params) => {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            const nodeData = nodes.get(nodeId);
            showNodeDetail(nodeData._raw);
        }
    });

    // Double-click to focus
    visNetwork.on('doubleClick', (params) => {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            visNetwork.focus(nodeId, { scale: 1.5, animation: true });
            loadSubgraph(nodeId);
        }
    });
}

// ============================================================
// 3D Point Cloud (3d-force-graph)
// ============================================================

function render3DGraph() {
    const container = document.getElementById('graph-3d');
    const filtered = filterGraphData();

    // Build lookup for quick access
    const nodeMap = {};
    filtered.nodes.forEach(n => { nodeMap[n.id] = n; });

    // Clean edges (only include edges where both nodes exist)
    const validEdges = filtered.edges.filter(e =>
        nodeMap[e.source] && nodeMap[e.target]
    );

    const data = {
        nodes: filtered.nodes.map(n => ({
            id: n.id,
            name: n.label || n.id,
            nodeType: n.node_type || 'Entity',
            val: (NODE_SIZES[n.node_type] || 10) / 5,
            color: NODE_COLORS[n.node_type] || '#6e7681',
            _raw: n,
        })),
        links: validEdges.map(e => ({
            source: e.source,
            target: e.target,
            relation: e.relation || 'RELATED',
            weight: e.weight || 0.5,
        })),
    };

    if (forceGraph3D) {
        // Update existing
        forceGraph3D.graphData(data);
        return;
    }

    forceGraph3D = ForceGraph3D()(container)
        .graphData(data)
        .nodeLabel(n => `${n.name} (${n.nodeType})`)
        .nodeColor(n => n.color)
        .nodeVal(n => n.val)
        .nodeOpacity(0.9)
        .linkColor(() => 'rgba(88, 166, 255, 0.2)')
        .linkWidth(l => Math.max(0.5, l.weight * 2))
        .linkOpacity(0.3)
        .linkDirectionalArrowLength(3)
        .linkDirectionalArrowRelPos(1)
        .backgroundColor('#0d1117')
        .onNodeClick(node => {
            showNodeDetail(node._raw);
            // Zoom to node
            const distance = 200;
            const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
            forceGraph3D.cameraPosition(
                { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
                node,
                1000
            );
        })
        .onNodeHover(node => {
            container.style.cursor = node ? 'pointer' : 'default';
        })
        .d3Force('charge', null)
        .d3Force('link', null)
        .d3Force('center', null);

    // Custom forces for clustering by type
    const d3 = forceGraph3D.d3Force;
    forceGraph3D
        .d3Force('charge', window.d3 ? window.d3.forceManyBody().strength(-50) : null)
        .warmupTicks(50)
        .cooldownTicks(100);
}

// ============================================================
// Timeline View
// ============================================================

async function renderTimeline() {
    const container = document.getElementById('timeline-content');
    try {
        const resp = await fetch('/api/timeline');
        const timeline = await resp.json();

        if (!timeline.length) {
            container.innerHTML = '<p style="color: var(--text-secondary)">No timeline data available.</p>';
            return;
        }

        let html = '<h2 style="margin-bottom: 24px; color: var(--text-primary)">Research Timeline</h2>';

        timeline.forEach(entry => {
            html += `
                <div class="timeline-year">
                    <div class="year-label">${entry.year}</div>
                    <div class="year-content">
                        <div style="color: var(--text-secondary); margin-bottom: 6px;">${entry.count} publication(s)</div>
                        <div class="timeline-domains">
                            ${Object.keys(entry.domains || {}).map(d =>
                                `<span class="tag domain">${d}</span>`
                            ).join('')}
                        </div>
                        ${entry.publications.map(p => `
                            <div class="timeline-pub" onclick="showPublicationDetail('${p.file_hash}')">
                                ${p.title}
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
    } catch (err) {
        container.innerHTML = '<p style="color: var(--red)">Failed to load timeline.</p>';
    }
}

// ============================================================
// Focused Views
// ============================================================

async function loadKeywordNetwork() {
    try {
        const resp = await fetch('/api/keywords');
        const data = await resp.json();
        graphData = {
            nodes: data.nodes.map(n => ({ ...n, node_type: 'Keyword' })),
            edges: data.edges,
        };
        populateFilters();
        renderCurrentView();
    } catch (err) { console.error(err); }
}

async function loadAuthorNetwork() {
    try {
        const resp = await fetch('/api/authors');
        const data = await resp.json();
        graphData = {
            nodes: data.nodes.map(n => ({ ...n, node_type: 'Author' })),
            edges: data.edges,
        };
        populateFilters();
        renderCurrentView();
    } catch (err) { console.error(err); }
}

async function loadConceptMap() {
    try {
        const resp = await fetch('/api/concepts');
        const data = await resp.json();
        graphData = {
            nodes: data.nodes.map(n => ({ ...n, node_type: 'Concept' })),
            edges: data.edges,
        };
        populateFilters();
        renderCurrentView();
    } catch (err) { console.error(err); }
}

async function loadClusters() {
    try {
        const resp = await fetch('/api/clusters');
        const clusters = await resp.json();

        // Build cluster visualization as a graph
        const nodes = [];
        const edges = [];

        clusters.forEach(cluster => {
            const domainId = `cluster_${cluster.domain}`;
            nodes.push({
                id: domainId,
                label: cluster.domain,
                node_type: 'Domain',
                size: cluster.size,
            });

            cluster.publications.forEach(pub => {
                const pubId = `cpub_${pub.file_hash}`;
                nodes.push({
                    id: pubId,
                    label: pub.title,
                    node_type: 'Publication',
                    file_hash: pub.file_hash,
                });
                edges.push({
                    source: pubId,
                    target: domainId,
                    relation: 'IN_CLUSTER',
                    weight: 0.7,
                });
            });
        });

        graphData = { nodes, edges };
        populateFilters();
        renderCurrentView();
    } catch (err) { console.error(err); }
}

async function loadSubgraph(nodeId) {
    try {
        const resp = await fetch(`/api/graph/subgraph/${encodeURIComponent(nodeId)}?depth=2`);
        const data = await resp.json();
        graphData = data;
        populateFilters();
        renderCurrentView();
    } catch (err) { console.error(err); }
}

// ============================================================
// Search
// ============================================================

async function doSearch() {
    const query = document.getElementById('search-input').value.trim();
    if (!query) return;

    try {
        const resp = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const results = await resp.json();
        renderSearchResults(results);

        // Highlight matching nodes in current view
        if (visNetwork && currentView === '2d') {
            const matchIds = results.graph_nodes.map(n => n.id);
            visNetwork.selectNodes(matchIds);
            if (matchIds.length === 1) {
                visNetwork.focus(matchIds[0], { scale: 1.5, animation: true });
            }
        }
    } catch (err) {
        console.error('Search failed:', err);
    }
}

function renderSearchResults(results) {
    const container = document.getElementById('search-results');
    let html = '';

    if (results.publications && results.publications.length) {
        results.publications.forEach(pub => {
            html += `
                <div class="result-item" onclick="showPublicationDetail('${pub.file_hash}')"
                     style="border-left-color: var(--color-publication)">
                    <div class="title">${pub.title}</div>
                    <div class="meta">${(pub.authors || []).join(', ')} ${pub.year ? '(' + pub.year + ')' : ''}</div>
                </div>
            `;
        });
    }

    if (results.graph_nodes && results.graph_nodes.length) {
        results.graph_nodes.slice(0, 10).forEach(node => {
            const color = NODE_COLORS[node.node_type] || '#6e7681';
            html += `
                <div class="result-item" onclick="focusNode('${node.id}')"
                     style="border-left-color: ${color}">
                    <div class="title">${node.label || node.id}</div>
                    <div class="meta">${node.node_type || 'Node'}</div>
                </div>
            `;
        });
    }

    if (!html) html = '<div class="result-item"><div class="meta">No results found</div></div>';
    container.innerHTML = html;
}

function focusNode(nodeId) {
    if (visNetwork && currentView === '2d') {
        visNetwork.selectNodes([nodeId]);
        visNetwork.focus(nodeId, { scale: 1.5, animation: true });
    }
    // Fetch and show detail
    showNodeDetailById(nodeId);
}

// ============================================================
// Detail Panel
// ============================================================

async function showPublicationDetail(fileHash) {
    try {
        const resp = await fetch(`/api/publication/${fileHash}`);
        if (!resp.ok) return;
        const data = await resp.json();
        const pub = data.publication;

        const relResp = await fetch(`/api/publication/${fileHash}/related`);
        const related = await relResp.json();

        let html = `
            <div class="detail-title">${pub.title}</div>
            <div style="color: var(--text-secondary); font-size: 0.9em;">
                ${pub.authors.join(', ')} ${pub.year ? '(' + pub.year + ')' : ''}
            </div>
            <div style="margin-top: 4px;">
                <span class="tag">${pub.type}</span>
            </div>

            <div class="detail-section">
                <h3>Summary</h3>
                <p style="font-size: 0.9em; line-height: 1.5;">${pub.summary}</p>
            </div>

            <div class="detail-section">
                <h3>Domains</h3>
                <div class="tag-list">
                    ${pub.domains.map(d => `<span class="tag domain">${d}</span>`).join('')}
                </div>
            </div>

            <div class="detail-section">
                <h3>Topics</h3>
                <div class="tag-list">
                    ${pub.topics.map(t => `<span class="tag topic">${t}</span>`).join('')}
                </div>
            </div>

            <div class="detail-section">
                <h3>Keywords</h3>
                <div class="tag-list">
                    ${pub.keywords.map(k => `<span class="tag keyword">${k}</span>`).join('')}
                </div>
            </div>

            <div class="detail-section">
                <h3>Methodologies</h3>
                <div class="tag-list">
                    ${pub.methodologies.map(m => `<span class="tag method">${m}</span>`).join('')}
                </div>
            </div>

            <div class="detail-section">
                <h3>Key Findings</h3>
                <ul style="font-size: 0.85em; padding-left: 16px; line-height: 1.5;">
                    ${pub.findings.map(f => `<li>${f}</li>`).join('')}
                </ul>
            </div>
        `;

        if (related && related.length) {
            html += `
                <div class="detail-section">
                    <h3>Related Publications</h3>
                    <ul class="related-list">
                        ${related.map(r => `
                            <li onclick="showPublicationDetail('${r.file_hash}')" style="cursor:pointer">
                                <span class="score">${r.score}</span> ${r.title}
                                <div style="font-size: 0.75em; color: var(--text-muted)">
                                    ${r.relation_types.join(', ')}
                                </div>
                            </li>
                        `).join('')}
                    </ul>
                </div>
            `;
        }

        // Manual link creation for publications too
        const pubNodeId = `pub_${fileHash.substring(0, 12)}`;
        html += buildAddLinkForm(pubNodeId);

        document.getElementById('detail-content').innerHTML = html;
        openDetail();
    } catch (err) {
        console.error('Failed to load publication detail:', err);
    }
}

async function showNodeDetail(node) {
    if (!node) return;

    // If it's a publication node, show full detail
    if (node.node_type === 'Publication' && node.file_hash) {
        showPublicationDetail(node.file_hash);
        return;
    }

    const nodeColor = NODE_COLORS[node.node_type] || '#6e7681';

    let html = `
        <div class="detail-title">${node.label || node.id}</div>
        <div style="color: var(--text-secondary);">
            <span class="tag" style="border-color: ${nodeColor}; color: ${nodeColor}">
                ${node.node_type || 'Node'}
            </span>
        </div>
    `;

    // Show all properties
    const skip = new Set(['id', 'label', 'node_type', 'title']);
    const props = Object.entries(node).filter(([k]) => !k.startsWith('_') && !skip.has(k));
    if (props.length) {
        html += '<div class="detail-section"><h3>Properties</h3>';
        props.forEach(([key, val]) => {
            html += `<div style="font-size: 0.85em; margin-bottom: 4px;">
                <strong>${key}:</strong> ${typeof val === 'object' ? JSON.stringify(val) : val}
            </div>`;
        });
        html += '</div>';
    }

    // Fetch linked publications
    try {
        const pubResp = await fetch(`/api/node/${encodeURIComponent(node.id)}/publications`);
        if (pubResp.ok) {
            const pubs = await pubResp.json();
            if (pubs.length) {
                html += '<div class="detail-section"><h3>Appears In</h3><ul class="related-list">';
                pubs.forEach(p => {
                    const clickAttr = p.file_hash
                        ? `onclick="showPublicationDetail('${p.file_hash}')" style="cursor:pointer"`
                        : `onclick="focusNode('${p.node_id}')" style="cursor:pointer"`;
                    const authors = p.authors ? p.authors.join(', ') : '';
                    const year = p.year ? ` (${p.year})` : '';
                    const meta = (authors || year)
                        ? `<div style="font-size:0.75em; color:var(--text-muted)">${authors}${year}</div>`
                        : '';
                    html += `<li ${clickAttr}>
                        ${p.title}
                        ${meta}
                        <span class="score">${p.relation || ''}</span>
                    </li>`;
                });
                html += '</ul></div>';
            }
        }
    } catch (err) { /* ignore */ }

    // Fetch neighbors (non-publication)
    try {
        const resp = await fetch(`/api/node/${encodeURIComponent(node.id)}`);
        if (resp.ok) {
            const data = await resp.json();
            const nonPubs = (data.neighbors || []).filter(n => n.node_type !== 'Publication');
            if (nonPubs.length) {
                html += '<div class="detail-section"><h3>Connected Nodes</h3><ul class="related-list">';
                nonPubs.forEach(n => {
                    const color = NODE_COLORS[n.node_type] || '#6e7681';
                    html += `<li onclick="focusNode('${n.id}')" style="cursor:pointer">
                        <span style="color:${color}">${n.node_type}</span>:
                        ${n.label || n.id}
                        <span class="score">${n._edge_relation || ''}</span>
                    </li>`;
                });
                html += '</ul></div>';
            }
        }
    } catch (err) { /* ignore */ }

    // Manual link creation form
    html += buildAddLinkForm(node.id);

    document.getElementById('detail-content').innerHTML = html;
    openDetail();
}

async function showNodeDetailById(nodeId) {
    try {
        const resp = await fetch(`/api/node/${encodeURIComponent(nodeId)}`);
        if (resp.ok) {
            const data = await resp.json();
            showNodeDetail(data.node);
        }
    } catch (err) { console.error(err); }
}

function openDetail() {
    const panel = document.getElementById('detail-panel');
    panel.classList.remove('hidden');
    panel.classList.add('visible');
}

function closeDetail() {
    const panel = document.getElementById('detail-panel');
    panel.classList.remove('visible');
    panel.classList.add('hidden');
}

// ============================================================
// Filters
// ============================================================

function populateFilters() {
    // Node type filters
    const typeContainer = document.getElementById('type-filters');
    const types = new Set(graphData.nodes.map(n => n.node_type || 'Entity'));
    activeNodeTypes = new Set(types);

    typeContainer.innerHTML = [...types].map(t => `
        <span class="filter-chip active" data-type="${t}" onclick="toggleTypeFilter(this, '${t}')">
            <span class="dot" style="background: ${NODE_COLORS[t] || '#6e7681'}"></span>
            ${t}
        </span>
    `).join('');

    // Edge type filters
    const edgeContainer = document.getElementById('edge-filters');
    const edgeTypes = new Set(graphData.edges.map(e => e.relation || 'RELATED'));
    activeEdgeTypes = new Set(edgeTypes);

    edgeContainer.innerHTML = [...edgeTypes].map(t => `
        <span class="filter-chip active" data-edge="${t}" onclick="toggleEdgeFilter(this, '${t}')">
            ${t.replace(/_/g, ' ')}
        </span>
    `).join('');
}

function toggleTypeFilter(el, type) {
    if (activeNodeTypes.has(type)) {
        activeNodeTypes.delete(type);
        el.classList.remove('active');
    } else {
        activeNodeTypes.add(type);
        el.classList.add('active');
    }
    renderCurrentView();
}

function toggleEdgeFilter(el, type) {
    if (activeEdgeTypes.has(type)) {
        activeEdgeTypes.delete(type);
        el.classList.remove('active');
    } else {
        activeEdgeTypes.add(type);
        el.classList.add('active');
    }
    renderCurrentView();
}

function filterGraphData() {
    const nodeIds = new Set();
    const nodes = graphData.nodes.filter(n => {
        const type = n.node_type || 'Entity';
        if (activeNodeTypes.has(type)) {
            nodeIds.add(n.id);
            return true;
        }
        return false;
    });

    const edges = graphData.edges.filter(e => {
        const rel = e.relation || 'RELATED';
        return activeEdgeTypes.has(rel) && nodeIds.has(e.source) && nodeIds.has(e.target);
    });

    return { nodes, edges };
}

// ============================================================
// Stats
// ============================================================

function renderStats(stats) {
    const container = document.getElementById('stats-panel');
    const db = stats.database || {};
    const graph = stats.graph || {};

    container.innerHTML = `
        <div class="stat-card">
            <div class="value">${db.publications || 0}</div>
            <div class="label">Publications</div>
        </div>
        <div class="stat-card">
            <div class="value">${graph.total_nodes || 0}</div>
            <div class="label">Graph Nodes</div>
        </div>
        <div class="stat-card">
            <div class="value">${graph.total_edges || 0}</div>
            <div class="label">Graph Edges</div>
        </div>
        <div class="stat-card">
            <div class="value">${db.cross_edges || 0}</div>
            <div class="label">Connections</div>
        </div>
    `;
}

// ============================================================
// Utilities
// ============================================================

function truncateLabel(text, maxLen) {
    if (!text) return '';
    return text.length > maxLen ? text.substring(0, maxLen) + '...' : text;
}

function buildTooltipHTML(node) {
    let html = `<strong>${node.label || node.id}</strong><br>`;
    html += `<em>${node.node_type || 'Node'}</em>`;
    if (node.year) html += `<br>Year: ${node.year}`;
    if (node.summary) html += `<br>${truncateLabel(node.summary, 100)}`;
    if (node.definition) html += `<br>${truncateLabel(node.definition, 100)}`;
    return html;
}

function lighten(hex) {
    const num = parseInt(hex.replace('#', ''), 16);
    const r = Math.min(255, ((num >> 16) & 0xFF) + 40);
    const g = Math.min(255, ((num >> 8) & 0xFF) + 40);
    const b = Math.min(255, (num & 0xFF) + 40);
    return `rgb(${r}, ${g}, ${b})`;
}

// ============================================================
// Manual Link Creation
// ============================================================

const RELATION_PRESETS = [
    'RELATED_CONCEPT', 'SAME_AS', 'SUBTOPIC_OF', 'EXTENDS',
    'CONTRASTS', 'ENABLES', 'REQUIRES', 'CO_AUTHORED',
    'SHARES_KEYWORDS', 'SHARES_TOPICS', 'APPLIES_TO',
];

function buildAddLinkForm(sourceNodeId) {
    const options = RELATION_PRESETS.map(r =>
        `<option value="${r}">${r.replace(/_/g, ' ')}</option>`
    ).join('');

    return `
        <div class="detail-section">
            <h3>Add Manual Link</h3>
            <div class="link-form" style="font-size: 0.85em;">
                <div style="margin-bottom: 6px;">
                    <label style="font-size: 0.8em; color: var(--text-secondary); text-transform: none; letter-spacing: normal;">
                        Target node (search)
                    </label>
                    <input type="text" id="link-target-search"
                        placeholder="Type to search nodes..."
                        oninput="searchLinkTarget(this.value)"
                        autocomplete="off"
                        style="width: 100%; padding: 6px 8px; margin-top: 2px;
                               background: var(--bg-tertiary); border: 1px solid var(--border);
                               border-radius: 4px; color: var(--text-primary); font-size: 1em;">
                    <div id="link-target-results" style="max-height: 120px; overflow-y: auto; margin-top: 2px;"></div>
                    <input type="hidden" id="link-target-id" value="">
                </div>
                <div style="margin-bottom: 6px;">
                    <label style="font-size: 0.8em; color: var(--text-secondary); text-transform: none; letter-spacing: normal;">
                        Relation type
                    </label>
                    <select id="link-relation"
                        style="width: 100%; padding: 6px 8px; margin-top: 2px;
                               background: var(--bg-tertiary); border: 1px solid var(--border);
                               border-radius: 4px; color: var(--text-primary); font-size: 1em;">
                        ${options}
                    </select>
                </div>
                <div id="link-form-msg" style="font-size: 0.8em; margin-bottom: 4px;"></div>
                <button class="btn btn-sm" onclick="submitManualLink('${sourceNodeId}')"
                    style="margin-top: 4px;">
                    Create Link
                </button>
            </div>
        </div>
    `;
}

let _linkSearchTimeout = null;

function searchLinkTarget(query) {
    clearTimeout(_linkSearchTimeout);
    const container = document.getElementById('link-target-results');
    if (!query || query.length < 2) {
        container.innerHTML = '';
        return;
    }
    _linkSearchTimeout = setTimeout(async () => {
        try {
            const resp = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
            if (!resp.ok) return;
            const results = await resp.json();
            const items = [
                ...(results.graph_nodes || []).slice(0, 8),
                ...(results.publications || []).slice(0, 4).map(p => ({
                    id: `pub_${p.file_hash.substring(0, 12)}`,
                    label: p.title,
                    node_type: 'Publication',
                })),
            ];
            if (!items.length) {
                container.innerHTML = '<div style="padding:4px; color:var(--text-muted); font-size:0.8em">No matches</div>';
                return;
            }
            container.innerHTML = items.map(n => {
                const color = NODE_COLORS[n.node_type] || '#6e7681';
                return `<div class="result-item" style="padding:4px 6px; border-left-color:${color}"
                    onclick="selectLinkTarget('${n.id}', '${(n.label || n.id).replace(/'/g, "\\'")}')">
                    <span style="color:${color}; font-size:0.8em">${n.node_type || 'Node'}</span>
                    ${n.label || n.id}
                </div>`;
            }).join('');
        } catch (err) { /* ignore */ }
    }, 250);
}

function selectLinkTarget(nodeId, label) {
    document.getElementById('link-target-id').value = nodeId;
    document.getElementById('link-target-search').value = label;
    document.getElementById('link-target-results').innerHTML = '';
}

async function submitManualLink(sourceId) {
    const targetId = document.getElementById('link-target-id').value;
    const relation = document.getElementById('link-relation').value;
    const msgEl = document.getElementById('link-form-msg');

    if (!targetId) {
        msgEl.innerHTML = '<span style="color:var(--red)">Select a target node first.</span>';
        return;
    }

    if (sourceId === targetId) {
        msgEl.innerHTML = '<span style="color:var(--red)">Cannot link a node to itself.</span>';
        return;
    }

    try {
        const resp = await fetch('/api/edge', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: sourceId, target: targetId, relation }),
        });
        const data = await resp.json();
        if (resp.ok) {
            msgEl.innerHTML = `<span style="color:var(--green)">Link created: ${data.relation}</span>`;
            // Refresh the graph to show the new edge
            loadFullGraph();
        } else {
            msgEl.innerHTML = `<span style="color:var(--red)">${data.error || 'Failed'}</span>`;
        }
    } catch (err) {
        msgEl.innerHTML = `<span style="color:var(--red)">Network error</span>`;
    }
}
