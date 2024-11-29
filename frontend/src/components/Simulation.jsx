import React, {useEffect, useState} from 'react';
import {Line} from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';
import bg from './bagckground.jpg';
import bg2 from './bagckground-modified.jpg';
import shuttle from './shuttle.png'
import proximaCentauri from './proximaCentauri.png'

// Register necessary components for Chart.js
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const inputs = {
    initial_population: {label: 'Initial Population', value: 1000},
    ship_capacity: {label: 'Ship Capacity', value: 20000},
    initial_resources: {label: 'Initial Resources', value: 200000},
    birth_rate: {label: 'Birth Rate (per 1000 people)', value: 9.4},
    death_rate: {label: 'Death Rate (per 1000 people)', value: 3.7},
    health_index: {label: 'Health Index', value: 100},
    resource_gen_rate: {label: 'Resource Generation Rate', value: 20000},
    lightspeed_fraction: {label: 'Speed of Ship (% of lightSpeed)', value: 0.0059},
    yearsToSimulate: {label: 'Years to Simulate', value: 100},
};

const Simulation = () => {
    const DISTANCE_TO_PROXIMA_CENTAURI = 40140000000000;
    const SPEED_OF_LIGHT_KM_HR = 1079251200;
    const [formData, setFormData] = useState(inputs);
    const [simulationData, setSimulationData] = useState([]);


    const handleChange = (e, key) => {
        const updatedData = {...formData};
        updatedData[key].value = e.target.value;
        setFormData(updatedData);
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        setSimulationData([])

        const backendData = Object.keys(formData).reduce((acc, key) => {
            acc[key] = parseFloat(formData[key].value);
            return acc;
        }, {});

        fetch('http://localhost:5001/initialize', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(backendData),
        })
            .then((response) => {
                if (!response.ok) throw new Error('Failed to initialize simulation');
                return response.json();
            })
            .then(() => {
                return fetch('http://localhost:5001/simulate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({years: parseInt(formData.yearsToSimulate.value)}),
                });
            })
            .then((response) => response.json())
            .then((data) => {
                console.log(data)
                setSimulationData(data);
            })
            .catch((error) => console.error('Simulation error:', error));
    };

    // Prepare data for the line charts
    const chartData = {
        population: {
            labels: simulationData.map((yearData) => yearData.year),
            datasets: [
                {
                    label: 'Population',
                    data: simulationData.map((yearData) => yearData.population),
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.4)',
                    fill: true,
                    tension: 0.1,
                },
                {
                    label: 'Capacity',
                    data: simulationData.map((yearData) => yearData.ship_capacity),
                    borderColor: '#CA3433',
                    backgroundColor: '#CA3433',
                    fill: true,
                    tension: 0.9,
                },
            ],
        },
        resources: {
            labels: simulationData.map((yearData) => yearData.year),
            datasets: [
                {
                    label: 'Resources',
                    data: simulationData.map((yearData) => yearData.resources),
                    borderColor: 'rgba(153, 102, 255, 1)',
                    backgroundColor: 'rgba(153, 102, 255, 0.4)',
                    fill: true,
                    tension: 0.1,
                },
            ],
        },
        distanceCovered: {
            labels: simulationData.map((yearData) => yearData.year),
            datasets: [
                {
                    label: 'Distance Covered (km)',
                    data: simulationData.map((yearData) => yearData.distance_covered),
                    borderColor: 'rgba(255, 159, 64, 1)',
                    backgroundColor: 'rgba(255, 159, 64, 0.4)',
                    fill: true,
                    tension: 0.1,
                },
            ],
        },
        status: {
            labels: simulationData.map((yearData) => yearData.year),
            datasets: [
                {
                    label: 'Status (Success=1, Running=0, Failure=-1)',
                    data: simulationData.map((yearData) => {
                        if (yearData.status === "Running") return 0
                        if (yearData.status === "Success") return 1
                        if (yearData.status === "Failed") return -1
                    }),
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.4)',
                    fill: true,
                    tension: 0.1,
                },
            ],
        },
        birthRate_deathRate: {
            labels: simulationData.map((yearData) => yearData.year),
            datasets: [
                {
                    label: 'Birth Rate',
                    data: simulationData.map((yearData) => yearData.birth_rate),
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.4)',
                    fill: true,
                    tension: 0.1,
                },
                {
                    label: 'Death Rate',
                    data: simulationData.map((yearData) => yearData.death_rate),
                    borderColor: 'rgba(255, 206, 86, 1)',
                    backgroundColor: 'rgba(255, 206, 86, 0.4)',
                    fill: true,
                    tension: 0.1,
                },
            ],
        },
        healthIndex: {
            labels: simulationData.map((yearData) => yearData.year),
            datasets: [
                {
                    label: 'Health Index',
                    data: simulationData.map((yearData) => yearData.health_index),
                    borderColor: 'rgba(255, 206, 86, 1)',
                    backgroundColor: 'rgba(255, 206, 86, 0.4)',
                    fill: true,
                    tension: 0.1,
                },
            ],
        },
    };

    const chartOptions = {
        responsive: true,
        plugins: {
            legend: {
                labels: {
                    color: '#FFFFFF', // Legend text color
                },
            }
        },
        scales: {
            x: {
                ticks: {
                    color: '#FFFFFF', // Tick color for x-axis
                },
                grid: {
                    color: '#444444', // Gridline color for x-axis
                },
            },
            y: {
                ticks: {
                    color: '#FFFFFF', // Tick color for y-axis
                },
                grid: {
                    color: '#444444', // Gridline color for y-axis
                },
            },
        },
    };

    return (
        <div style={{
            backgroundColor: '#121212',
            color: '#FFFFFF',
            minHeight: '100vh',
            padding: '20px',
            backgroundImage: `url(${simulationData.length > 0 ? bg2 : bg})`
        }}>
            <div style={{display: 'flex', margin: '0 auto', opacity: 0.90, height:'96vh'}}>
                <div style={{padding: 20,paddingTop:5, width: '22vw', backgroundColor: '#1e1e1e', borderRadius: 10}}>
                    <h2 style={{color: '#E0E0E0', width: '100%', textAlign: 'center'}}>Generation Ship Simulation</h2>
                    <form onSubmit={handleSubmit} style={{textAlign: 'center'}}>
                        {Object.keys(formData).map((key) => (
                            <>
                                <div key={key} style={{margin: 10, display: "flex"}}>
                                    <h5 style={{
                                        margin: 10,
                                        color: '#B0B0B0',
                                        width: '55%',
                                        verticalAlign: 'center',
                                        height: '100%'
                                    }}>{formData[key].label}:</h5>
                                    <>
                                        <input
                                            type="number"
                                            value={formData[key].value}
                                            onChange={(e) => handleChange(e, key)}
                                            style={{
                                                width: '30%',
                                                borderRadius: 5,
                                                border: '1px solid #444',
                                                backgroundColor: '#2b2b2b',
                                                color: '#FFFFFF',
                                                textAlign: 'center'
                                            }}
                                        />
                                    </>
                                </div>
                            </>
                        ))}
                        <div style={{margin: 5, padding: 5, backgroundColor: '#1e1e1e', borderRadius: 10}}>
                            <h5 style={{margin: 5, padding: 5, color: '#B0B0B0'}}>Distance
                                : {DISTANCE_TO_PROXIMA_CENTAURI} km
                            </h5>
                            <h5 style={{margin: 5, padding: 5, color: '#B0B0B0'}}>Speed of
                                Ship
                                : {(formData['lightspeed_fraction']['value'] * SPEED_OF_LIGHT_KM_HR).toFixed(2)} km/hr
                            </h5>
                            <h5 style={{margin: 5, padding: 5, color: '#B0B0B0'}}>Total yrs of travel
                                : {((DISTANCE_TO_PROXIMA_CENTAURI /
                                    (formData['lightspeed_fraction']['value'] * SPEED_OF_LIGHT_KM_HR)) / (24 * 365)).toFixed(2)}{' '}
                                yrs
                            </h5>
                        </div>
                        <button
                            type="submit"
                            style={{
                                margin: 10,
                                padding: 10,
                                cursor: 'pointer',
                                backgroundColor: '#4CAF50',
                                color: '#FFFFFF',
                                border: 'none',
                                borderRadius: 5,
                            }}
                        >
                            Run Simulation
                        </button>
                    </form>
                </div>

                <div style={{width: '80vw', opacity: 0.90,height:'96vh'}}>
                    {simulationData.length > 0 ? <div style={{position: "relative", height: 30, background: "#F2F3F2", borderRadius: "20px", overflow: "hidden", margin: 10}}>
                        <div
                            style={{
                                width: `${(simulationData.slice(-1)[0]['distance_covered'] / DISTANCE_TO_PROXIMA_CENTAURI) * 98}%`,
                                height: "100%",
                                background: "linear-gradient(to right, #8c8c8c, #F2F3F2)",
                                transition: "width 0.5s ease",
                            }}
                        ></div>
                            {(simulationData.slice(-1)[0]['distance_covered'] / DISTANCE_TO_PROXIMA_CENTAURI) * 100>5?<label style={{
                            height: 20,
                            position: "absolute",
                            top: 4,
                            marginLeft: `${(simulationData.slice(-1)[0]['distance_covered'] / DISTANCE_TO_PROXIMA_CENTAURI) * 50}%`,
                            marginRight: 10,
                            transform: "translateX(-50%)",
                            transition: "left 0.5s ease",
                            color: '#1e1e1e',
                                fontWeight:"bold"
                        }}>{parseInt((simulationData.slice(-1)[0]['distance_covered'] / DISTANCE_TO_PROXIMA_CENTAURI) * 100)} %</label>:<></>}
                        <img
                            src={shuttle}
                            alt="Space Shuttle"
                            style={{
                                height: 20,
                                position: "absolute",
                                top: 4,
                                left: `${(simulationData.slice(-1)[0]['distance_covered'] / DISTANCE_TO_PROXIMA_CENTAURI) * 98}%`,
                                transform: "translateX(-50%)",
                                transition: "left 0.5s ease",
                            }}
                        />
                        <img
                            src={proximaCentauri}
                            alt="Proxima"
                            style={{
                                height: 30,
                                position: "absolute",
                                top: 0,
                                left: `99%`,
                                transform: "translateX(-50%)",
                                transition: "left 0.5s ease",
                            }}
                        />

                    </div> : <></>}

                    <div style={{margin: 10, color: '#E0E0E0', width:'92%'}}>
                        {simulationData.length > 0 && (<>
                            <div style={{width: '100%', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, borderRadius: 10}}>
                                <div style={{padding: 15, backgroundColor: '#1e1e1e', borderRadius: 10}}>
                                <h5 style={{margin: 0}}>Population</h5>
                                    <Line data={chartData.population} options={chartOptions}/>
                                </div>
                                <div style={{padding: 15, backgroundColor: '#1e1e1e', borderRadius: 10}}>
                                    <h5 style={{margin: 0}}>Resources</h5>
                                    <Line data={chartData.resources} options={chartOptions}/>
                                </div>
                                {/*<div style={{padding: 15, backgroundColor: '#1e1e1e', borderRadius: 10}}>*/}
                                {/*    <h5 style={{margin: 0}}>Distance Covered</h5>*/}
                                {/*    <Line data={chartData.distanceCovered} options={chartOptions}/>*/}
                                {/*</div>*/}
                                <div style={{padding: 15, backgroundColor: '#1e1e1e', borderRadius: 10}}>
                                    <h5 style={{margin: 0}}>Birth Rate vs Death Rate</h5>
                                    <Line data={chartData.birthRate_deathRate} options={chartOptions}/>
                                </div>
                                <div style={{padding: 15, backgroundColor: '#1e1e1e', borderRadius: 10}}>
                                    <h5 style={{margin: 0}}>Health Index</h5>
                                    <Line data={chartData.healthIndex} options={chartOptions}/>
                                </div>
                                {/*<div style={{padding: 15, backgroundColor: '#1e1e1e', borderRadius: 10}}>*/}
                                {/*    <h5 style={{margin: 0}}>Status (Success/Failure)</h5>*/}
                                {/*    <Line data={chartData.status} options={chartOptions}/>*/}
                                {/*</div>*/}
                            </div>
                        </>)}
                    </div>
                    <div style={{display: 'flex', height:'40%'}}>
                        {simulationData.length > 0 ?
                            <div style={{margin: 10, padding: 15, backgroundColor: '#1e1e1e', borderRadius: 10,}}>
                                <h5 style={{color: '#B0B0B0', margin: 5}}>Simulation Logs:</h5>
                                <div style={{overflowY: 'auto', height: '85%', marginTop: 10}}>
                                    {simulationData.slice().reverse().map((entry, index) => (
                                        <>
                                            {index === 0 ? <div
                                                key={index}
                                                style={{
                                                    color: '#FFFFFF',
                                                    borderBottom: '1px solid #333',
                                                }}
                                            >
                                                <h4 style={{margin: 5}}>Mission Status :
                                                    <label style={{ borderRadius: 5, margin:5,padding: 5,backgroundColor:entry.status=='Failed'?'#CA3433':entry.status=='Running'?'rgba(54, 162, 235, 1)':'#4CAF50'}}>{entry.status}</label>
                                                </h4>
                                            </div> : <></>}
                                            <h4 style={{margin: 5}}>Year : {entry.year}</h4>
                                            <div
                                                key={index}
                                                style={{
                                                    color: '#FFFFFF',
                                                    padding: '5px',
                                                    borderBottom: '1px solid #333',
                                                    textWrap: 'wrap'
                                                }}
                                            >
                                                {entry.log}
                                            </div>
                                        </>
                                    ))}
                                </div>
                            </div> : <></>}
                    </div>
                </div>

            </div>
        </div>
    );
};

export default Simulation;
