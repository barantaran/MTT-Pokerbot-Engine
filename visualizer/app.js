let playbackState = {
    events: [],
    results: [],
    currentTime: 0,
    maxTime: 0,
    isPlaying: false,
    speed: 1,
    lastUpdate: 0,
    zoomedTableId: null,
    
    // Derived state
    players: {}, // name -> {stack, table_id, active, action}
    tables: {}, // id -> {board, pot, players_at_table[]}
    blinds: {small: 0, big: 0},
    aliveCount: 0,
    currentEventIndex: 0
};

// DOM Elements
const timeline = document.getElementById('timeline');
const playBtn = document.getElementById('btn-play-pause');
const timeDisplay = document.getElementById('time-display');
const leaderboardList = document.getElementById('leaderboard-list');
const arena = document.getElementById('arena');
const zoomOutBtn = document.getElementById('btn-zoom-out');
const statsAlive = document.getElementById('players-alive');
const statsBlinds = document.getElementById('current-blinds');

// Handle File Load
document.getElementById('sim-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    document.getElementById('loaded-filename').innerText = file.name;
    
    const reader = new FileReader();
    reader.onload = (ev) => {
        try {
            const data = JSON.parse(ev.target.result);
            initSimulation(data);
        } catch (err) {
            alert('Invalid JSON file.');
        }
    };
    reader.readAsText(file);
});

function initSimulation(data) {
    if (!data.events || data.events.length === 0) {
        alert("No events found in simulation file.");
        return;
    }
    
    playbackState.events = data.events;
    playbackState.results = data.results;
    
    // We'll assign an artificial 500ms delay to each event for smooth timeline playback
    playbackState.maxTime = data.events.length * 500;
    playbackState.currentTime = 0;
    playbackState.currentEventIndex = 0;
    playbackState.zoomedTableId = null;
    
    timeline.max = playbackState.maxTime;
    timeline.value = 0;
    timeline.disabled = false;
    playBtn.disabled = false;
    
    resetDerivedState();
    updateView();
}

function resetDerivedState() {
    playbackState.players = {};
    playbackState.tables = {};
    playbackState.blinds = {small: 0, big: 0};
    playbackState.aliveCount = 0;
}

// ----------------------------------------------------
// Event Parsers
// ----------------------------------------------------

function applyEvent(ev) {
    switch (ev.type) {
        case "tournament_start":
            playbackState.blinds = ev.blinds;
            ev.players.forEach(p => {
                playbackState.players[p] = { name: p, stack: 1500 /* fallback, should be from hand_start */, table_id: null, active: true, action: "" };
                playbackState.aliveCount++;
            });
            break;
        case "seat":
            if (!playbackState.players[ev.player]) {
                 playbackState.players[ev.player] = { name: ev.player, stack: 1500, table_id: ev.table_id, active: true, action: "" };
            } else {
                 playbackState.players[ev.player].table_id = ev.table_id;
            }
            if (!playbackState.tables[ev.table_id]) {
                playbackState.tables[ev.table_id] = { board: [], pot: 0, action_text: "" };
            }
            break;
        case "table_broken":
            delete playbackState.tables[ev.table_id];
            break;
        case "level_up":
            playbackState.blinds = ev.blinds;
            break;
        case "hand_start":
            // Reset table state
            if (playbackState.tables[ev.table_id]) {
                playbackState.tables[ev.table_id].board = [];
                playbackState.tables[ev.table_id].pot = 0;
                playbackState.tables[ev.table_id].action_text = "New Hand";
            }
            // Update stacks
            ev.players.forEach(pData => {
                if (playbackState.players[pData.name]) {
                    playbackState.players[pData.name].stack = pData.stack;
                    playbackState.players[pData.name].action = "";
                    playbackState.players[pData.name].hole_cards = [];
                }
            });
            break;
        case "deal":
            if (playbackState.players[ev.player]) {
                playbackState.players[ev.player].hole_cards = ev.cards;
            }
            break;
        case "post_blind":
        case "action":
            if (playbackState.players[ev.player]) {
                playbackState.players[ev.player].action = `${ev.action || 'Blind'} ${ev.amount > 0 ? ev.amount : ''}`;
                playbackState.players[ev.player].stack -= ev.amount;
                if (playbackState.tables[ev.table_id]) {
                    playbackState.tables[ev.table_id].action_text = `${ev.player} ${ev.action || 'Blind'}s`;
                }
            }
            break;
        case "board":
            if (playbackState.tables[ev.table_id]) {
                playbackState.tables[ev.table_id].board.push(...ev.cards);
                playbackState.tables[ev.table_id].action_text = `Deal ${ev.street}`;
            }
            break;
        case "award_pot":
            if (playbackState.tables[ev.table_id]) {
                playbackState.tables[ev.table_id].pot += ev.amount; // Simplify pot track rendering
            }
            if (playbackState.players[ev.player]) {
                playbackState.players[ev.player].stack += ev.amount;
                playbackState.players[ev.player].action = `Won ${ev.amount}!`;
            }
            break;
        case "knockout":
            if (playbackState.players[ev.player]) {
                playbackState.players[ev.player].active = false;
                playbackState.players[ev.player].stack = 0;
                playbackState.players[ev.player].action = "Busted";
                playbackState.aliveCount = Math.max(0, playbackState.aliveCount - 1);
            }
            break;
        case "tournament_win":
            // Final screen
            document.getElementById('results-modal').classList.remove('hidden');
            renderResultsModal();
            break;
    }
}

function processUpToTime(timeMs) {
    const targetIdx = Math.floor(timeMs / 500);
    const maxIdx = Math.min(targetIdx, playbackState.events.length - 1);
    
    if (maxIdx < playbackState.currentEventIndex - 1) {
        // Scrubbed backwards - reset and fast forward
        resetDerivedState();
        playbackState.currentEventIndex = 0;
    }
    
    while (playbackState.currentEventIndex <= maxIdx && playbackState.currentEventIndex < playbackState.events.length) {
        applyEvent(playbackState.events[playbackState.currentEventIndex]);
        playbackState.currentEventIndex++;
    }
}

// ----------------------------------------------------
// Rendering
// ----------------------------------------------------

function updateView() {
    processUpToTime(playbackState.currentTime);
    
    // Update Header
    statsAlive.innerText = `Players: ${playbackState.aliveCount}`;
    statsBlinds.innerText = `Blinds: ${playbackState.blinds.small}/${playbackState.blinds.big}`;
    
    // Leaderboard
    const sortedPlayers = Object.values(playbackState.players).sort((a,b) => b.stack - a.stack);
    leaderboardList.innerHTML = sortedPlayers.map((p, idx) => `
        <div class="lb-player ${p.active ? '' : 'busted'}">
            <span class="lb-rank">#${idx + 1}</span>
            <span class="lb-name" title="${p.name}">${p.name}</span>
            <span class="lb-stack">${p.stack}</span>
            ${p.active && p.table_id ? `<span class="lb-table">T${p.table_id}</span>` : ''}
        </div>
    `).join('');
    
    // Arena
    if (playbackState.zoomedTableId === null) {
        renderAllTables();
    } else {
        renderZoomedTable(playbackState.zoomedTableId);
    }
    
    // Timeline text
    function formatTime(ms) {
        const totalSec = Math.floor(ms / 1000);
        const m = Math.floor(totalSec / 60);
        const s = totalSec % 60;
        return `${m}:${s.toString().padStart(2, '0')}`;
    }
    timeDisplay.innerText = `${formatTime(playbackState.currentTime)} / ${formatTime(playbackState.maxTime)}`;
    
    // Sync slider only if not currently dragging
    if (document.activeElement !== timeline) {
        timeline.value = playbackState.currentTime;
    }
}

function renderAllTables() {
    arena.innerHTML = '';
    zoomOutBtn.classList.add('hidden');
    
    const tables = Object.keys(playbackState.tables).map(Number).sort((a,b)=>a-b);
    if (tables.length === 0 && playbackState.events.length === 0) {
        arena.innerHTML = '<div id="welcome-msg">Load a JSON simulation log from /logs/ to begin playback.</div>';
        return;
    }
    
    tables.forEach(tId => {
        // Count active players at this table
        const pCount = Object.values(playbackState.players).filter(p => p.table_id === tId && p.active).length;
        if (pCount === 0) return; // Hide empty tables
        
        const div = document.createElement('div');
        div.className = 'poker-table-icon';
        div.innerHTML = `
            <div class="pt-id">Table ${tId}</div>
            <div class="pt-count">${pCount} Players</div>
        `;
        div.onmousedown = () => {
            playbackState.zoomedTableId = tId;
            updateView();
        };
        arena.appendChild(div);
    });
}

function renderZoomedTable(tId) {
    zoomOutBtn.classList.remove('hidden');
    const tableData = playbackState.tables[tId];
    if (!tableData) {
        playbackState.zoomedTableId = null;
        renderAllTables();
        return;
    }
    
    const playersAtTable = Object.values(playbackState.players).filter(p => p.table_id === tId && p.active);
    
    // Helper to render string cards (e.g. "As", "Th") into HTML playing cards
    const renderCard = (cardStr, isBack = false) => {
        if (isBack) return `<div class="card back"></div>`;
        if (!cardStr) return `<div class="card black">*</div>`; // Fallback
        
        let rank = cardStr[0].toUpperCase();
        let suit = cardStr[1].toLowerCase();
        let suitSymbol = '';
        let colorClass = 'black';
        
        // Handle "T" for ten
        if (rank === 'T') rank = '10';
        
        switch (suit) {
            case 's': suitSymbol = '♠'; break;
            case 'c': suitSymbol = '♣'; break;
            case 'h': suitSymbol = '♥'; colorClass = 'red'; break;
            case 'd': suitSymbol = '♦'; colorClass = 'red'; break;
            default: suitSymbol = '?';
        }
        
        return `<div class="card ${colorClass}">
                    <span>${rank}</span>
                    <span style="font-size: 0.8em; margin-left:2px;">${suitSymbol}</span>
                </div>`;
    };
    
    const boardHtml = tableData.board.map(c => renderCard(c)).join('');
    
    let seatsHtml = '';
    const radius = window.innerWidth < 800 ? 150 : 350;
    
    playersAtTable.forEach((p, index) => {
        const angle = (index / playersAtTable.length) * 2 * Math.PI - Math.PI/2;
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * (radius * 0.55); // Oval ratio
        
        let holeCardsHtml = '';
        if (p.hole_cards && p.hole_cards.length === 2) {
            holeCardsHtml = `
                <div class="hole-cards">
                    ${renderCard(p.hole_cards[0])}
                    ${renderCard(p.hole_cards[1])}
                </div>
            `;
        }
        
        seatsHtml += `
            <div class="seat" style="left: calc(50% + ${x}px); top: calc(50% + ${y}px);">
                <div class="name">${p.name}</div>
                <div class="chips">${p.stack}</div>
                <div class="action">${p.action}</div>
                ${holeCardsHtml}
            </div>
        `;
    });
    
    arena.innerHTML = `
        <div class="zoomed-table">
            <div class="board-cards">
                ${boardHtml || `<div style="color:var(--text-secondary);align-self:center;font-style:italic;">Preflop</div>`}
            </div>
            <div class="pot-total">Pot: ${tableData.pot}</div>
            <div style="position: absolute; bottom: 20px; color: var(--accent);">${tableData.action_text || ''}</div>
            ${seatsHtml}
        </div>
    `;
}

function renderResultsModal() {
    const list = document.getElementById('results-list');
    list.innerHTML = playbackState.results.map((r, i) => `
        <div class="result-item">
            <span class="result-rank">${i+1}</span>
            <span class="result-name">${r.name} <small>(${r.bot_class})</small></span>
            <span class="result-payout">${(r.payout_pct * 100).toFixed(1)}%</span>
        </div>
    `).join('');
}


// ----------------------------------------------------
// UI Controls Binding
// ----------------------------------------------------

zoomOutBtn.onmousedown = () => {
    playbackState.zoomedTableId = null;
    updateView();
};

document.getElementById('btn-close-modal').onclick = () => {
    document.getElementById('results-modal').classList.add('hidden');
};

timeline.addEventListener('input', (e) => {
    playbackState.currentTime = parseInt(e.target.value);
    updateView();
});

playBtn.addEventListener('click', () => {
    playbackState.isPlaying = !playbackState.isPlaying;
    playBtn.innerText = playbackState.isPlaying ? '⏸' : '▶';
    if (playbackState.isPlaying) {
        playbackState.lastUpdate = performance.now();
        requestAnimationFrame(tick);
    }
});

document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        playbackState.speed = parseInt(e.target.dataset.speed);
    });
});

function tick(now) {
    if (!playbackState.isPlaying) return;
    
    const dt = now - playbackState.lastUpdate;
    playbackState.lastUpdate = now;
    
    playbackState.currentTime += (dt * playbackState.speed);
    
    if (playbackState.currentTime >= playbackState.maxTime) {
        playbackState.currentTime = playbackState.maxTime;
        playbackState.isPlaying = false;
        playBtn.innerText = '▶';
    }
    
    updateView();
    
    if (playbackState.isPlaying) {
        requestAnimationFrame(tick);
    }
}
