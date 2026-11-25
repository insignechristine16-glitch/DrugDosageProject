const $ = id => document.getElementById(id);
const STATUS = {'OK+': 's-okplus', OK: 's-ok', low: 's-low', dip: 's-dip', high: 's-high', swing: 's-swing', 'n/c': 's-nc'};
const REFERENCE = JSON.parse(document.querySelector('main').dataset.reference);
const PRESETS = {   // the study's experiments: graph length, then [mg/day, doses/day, volume]
  split: [48, [1, 2, 3, 4, 6].map(doses => [6000, doses, 4.6])],
  totals: [72, [6000, 9000, 12000, 15000].map(daily => [daily, 4, 4.6])],
  volumes: [72, [3.68, 4.6, 5.52].map(volume => [REFERENCE[0], REFERENCE[1], volume])],
};
const LEGEND = [['OK+', 'in range from first entry to the end'], ['OK', 'repeating cycle in range, dips during build-up'],
  ['low', 'never reaches the medicinal level'], ['dip', 'troughs fall below it'], ['high', 'peaks reach the toxic level'],
  ['swing', 'both limits crossed'], ['n/c', 'not settled yet']];

async function ask(path, query) {
  try {
    return await (await fetch(path + '?' + new URLSearchParams(query))).json();
  } catch (error) {
    return {found: false, headline: 'The calculator has stopped', details: ['Run the Python file again to restart it.']};
  }
}

function fillList(list, items) {
  list.replaceChildren(...items.map(text => {
    const item = document.createElement('li');
    item.textContent = text;
    return item;
  }));
}

function fillTable(table, headers, rows, statusFrom = Infinity) {
  table.replaceChildren();
  if (headers.length) {
    const head = table.createTHead().insertRow();
    headers.forEach(text => {
      const cell = document.createElement('th');
      cell.scope = 'col';
      cell.textContent = text;
      head.append(cell);
    });
  }
  const body = table.createTBody();
  rows.forEach(values => {
    const row = body.insertRow();
    values.forEach((text, index) => {
      const cell = row.insertCell();
      cell.textContent = text;
      if (index >= statusFrom) cell.className = STATUS[text] || '';
    });
  });
}

function showAnswer(prefix, answer) {
  const headline = $(prefix + '-headline');
  headline.textContent = answer.headline;
  headline.classList.toggle('problem', !answer.found);
  fillList($(prefix + '-details'), answer.details || []);
  const graph = $(prefix + '-graph');
  graph.hidden = !answer.graph;
  if (answer.graph) graph.src = 'data:image/png;base64,' + answer.graph;
}

function connect(formId, path, show) {    // a form that sends its fields to the program and shows the answer
  const form = $(formId), button = form.querySelector('button[type=submit]'), label = button.textContent;
  let latestRun = 0;
  const run = async event => {
    if (event) event.preventDefault();
    const runId = ++latestRun;
    button.disabled = true;
    button.textContent = 'Calculating...';
    const answer = await ask(path, Object.fromEntries(new FormData(form)));
    if (runId !== latestRun) return;
    show(answer);
    button.disabled = false;
    button.textContent = label;
  };
  form.addEventListener('submit', run);
  return run;
}

const runPatient = connect('form-patient', '/api/dose', answer => showAnswer('p', answer));

const runSchedule = connect('form-schedule', '/api/schedule', answer => {
  showAnswer('s', answer);
  const chip = $('s-status');
  chip.textContent = answer.status || '';
  chip.className = 'chip ' + (STATUS[answer.status] || '');
  chip.hidden = !answer.status;
  fillTable($('s-facts'), [], answer.facts || []);
  $('s-values-title').textContent = answer.values_title || 'Values over time';
  fillTable($('s-values'), answer.value_columns || [], answer.value_rows || []);
});

const compareForm = $('form-compare');
for (let i = 0; i < 5; i++) {
  const letter = 'ABCDE'[i];
  const row = $('c-row-template').content.cloneNode(true);
  row.querySelector('td').textContent = letter;
  row.querySelectorAll('[name]').forEach(field => {
    field.name += i;
    field.setAttribute('aria-label', `Schedule ${letter} ${field.dataset.label}`);
  });
  $('c-rows').append(row);
}
const runCompare = connect('form-compare', '/api/compare', answer => {
  showAnswer('c', answer);
  const columns = answer.columns || [];
  fillTable($('c-table'), columns, answer.rows || [], columns.length - 1);
});
function usePreset(name) {
  const [span, rows] = PRESETS[name];
  for (let i = 0; i < 5; i++) {
    const [daily, doses, volume] = rows[i] || ['', 4, ''];
    compareForm.elements['daily' + i].value = daily;
    compareForm.elements['doses' + i].value = doses;
    compareForm.elements['volume' + i].value = volume;
  }
  $('c-span').value = span;
  runCompare();
}
document.querySelectorAll('[data-preset]').forEach(button =>
  button.addEventListener('click', () => usePreset(button.dataset.preset)));

const runFind = connect('form-find', '/api/find', answer => {
  showAnswer('f', answer);
  fillTable($('f-map'), answer.map_columns || [], answer.map_rows || [], 1);
  $('f-refinement-note').textContent = answer.refinement_note || '';
  fillTable($('f-refinement'), answer.refinement_columns || [], answer.refinement_rows || [], 2);
  fillTable($('f-windows'), answer.window_columns || [], answer.window_rows || []);
});

const runMeasured = connect('form-measured', '/api/measured', answer => {
  showAnswer('m', answer);
  fillTable($('m-overall'), [], answer.overall || []);
  fillTable($('m-samples'), answer.sample_columns || [], answer.sample_rows || []);
  fillTable($('m-people'), answer.people_columns || [], answer.people_rows || []);
});
$('m-subject').addEventListener('change', runMeasured);

document.querySelectorAll('.legend').forEach(note => {
  note.innerHTML = LEGEND.map(([label, text]) => `<span class="chip ${STATUS[label]}">${label}</span> ${text}`).join(' &nbsp; ');
});

const tabs = [...document.querySelectorAll('[role=tab]')];
const firstOpen = {patient: runPatient, schedule: runSchedule, compare: () => usePreset('split'),
                   find: runFind, measured: runMeasured};
function openTab(name) {
  tabs.forEach(tab => {
    const open = tab.dataset.tab === name;
    tab.setAttribute('aria-selected', open);
    tab.tabIndex = open ? 0 : -1;
  });
  document.querySelectorAll('.tab').forEach(panel => { panel.hidden = panel.id !== 'tab-' + name; });
  history.replaceState(null, '', '#' + name);
  if (firstOpen[name]) {      // each calculation runs once by itself the first time its tab opens
    firstOpen[name]();
    delete firstOpen[name];
  }
}
tabs.forEach(tab => tab.addEventListener('click', () => openTab(tab.dataset.tab)));
$('tabs').addEventListener('keydown', event => {
  const move = {ArrowRight: 1, ArrowLeft: -1}[event.key];
  if (!move) return;
  const next = tabs[(tabs.indexOf(document.activeElement) + move + tabs.length) % tabs.length];
  next.focus();
  openTab(next.dataset.tab);
});
openTab(tabs.some(tab => '#' + tab.dataset.tab === location.hash) ? location.hash.slice(1) : 'patient');

$('stop').addEventListener('click', async () => {
  await fetch('/api/stop', {method: 'POST'}).catch(() => {});
  document.querySelector('main').innerHTML = '<h1>The calculator has stopped</h1>' +
    '<p class="lead">You can close this tab. Run the Python file again to use the calculator.</p>';
});
