// This code runs AFTER index.html is loaded and AFTER data.js
// has provided the 'embeddedData' variable.

console.log("app.js loaded.");
console.log("1. Raw embedded data:", embeddedData);

// --- Get HTML Elements ---
const plotDiv = document.getElementById('plot-div');
const sampleSelect = document.getElementById('sample-select');
const modal = document.getElementById('info-modal');
const modalText = document.getElementById('modal-text');
const closeButton = document.getElementsByClassName('close-button')[0];

// --- Helper Functions ---

/**
 * Creates a Plotly trace object from an array of data points.
 * @param {Array} dataPoints - An array of data objects.
 * @returns {Object} A Plotly trace object.
 */
function createPlotlyTrace(dataPoints) {
    return {
        x: dataPoints.map(d => d.x_pos),
        y: dataPoints.map(d => d.y_pos),
        mode: 'markers',
        type: 'scatter',
        name: 'Sample Data',
        marker: { 
            size: 15,
            color: '#1f77b4',
            opacity: 0.8
        },
        customdata: dataPoints,
        hovertemplate: '<b>%{customdata.time_stamp}</b><br>MAG: %{customdata.magnification}<extra></extra>'
    };
}

/**
 * Populates the dropdown menu with unique sample names.
 */
function populateDropdown() {
    // Get unique sample names from the data
    const sampleNames = [...new Set(embeddedData.map(d => d.sample_name))];
    
    // Add an "All Samples" option to the beginning
    const allOption = document.createElement('option');
    allOption.value = 'all';
    allOption.textContent = 'All Samples';
    sampleSelect.appendChild(allOption);

    // Add an option for each unique sample name
    sampleNames.forEach(name => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        sampleSelect.appendChild(option);
    });
}

/**
 * Updates the plot based on the selected sample name.
 */
function updatePlot() {
    const selectedSample = sampleSelect.value;
    console.log(`Dropdown changed. Selected sample: ${selectedSample}`);

    let filteredData;
    if (selectedSample === 'all') {
        filteredData = embeddedData; // Show all data
    } else {
        filteredData = embeddedData.filter(d => d.sample_name === selectedSample);
    }

    // Create a new trace with the filtered data
    const newTrace = createPlotlyTrace(filteredData);
    
    // Use Plotly.react() to efficiently update the plot
    Plotly.react(plotDiv, [newTrace], layout);
    console.log(`Plot updated with ${filteredData.length} points.`);
}

// --- Plot Layout & Config (Global) ---
const layout = {
    title: 'Interactive Scatter Plot',
    xaxis: { title: 'X Axis pos (um)' },
    yaxis: { title: 'Y Axis pos (um)' },
    hovermode: 'closest',
};

const config = {
    responsive: true,
    displayModeBar: false
};

// --- Initialization ---

function init() {
    // 1. Fill the dropdown
    populateDropdown();
    
    // 2. Create the initial trace with all data
    const initialTrace = createPlotlyTrace(embeddedData);
    
    // 3. Render the initial plot
    Plotly.newPlot(plotDiv, [initialTrace], layout, config);
    console.log("3. Plotly.newPlot() called.");

    // --- Event Listeners ---

    // 4. Add event listener for the dropdown
    sampleSelect.addEventListener('change', updatePlot);

    // 5. Add event listeners for the modal
    closeButton.onclick = function() {
        modal.style.display = 'none';
    }

    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    }

    // 6. Add click listener to the plot
    plotDiv.on('plotly_click', function(data) {
        console.log("4. Plotly click event fired:", data);
        
        const point = data.points[0];
        const pointInfo = point.customdata; 

        console.log("5. Clicked point info:", pointInfo);

        // Build the text to show in the modal
        let infoHtml = `
            <h2>${pointInfo.time_stamp}</h2>
            <img src="${pointInfo.image_data_uri}" alt="${pointInfo.time_stamp}">
            <hr>
            <p><strong>X-Value:</strong> ${pointInfo.x_pos}</p>
            <p><strong>Y-Value:</strong> ${pointInfo.y_pos}</p>
            <p><strong>Magnification:</strong> ${pointInfo.magnification}</p>
        `;

        modalText.innerHTML = infoHtml;
        modal.style.display = 'flex';
    });
}

// Run the initialization function
init();