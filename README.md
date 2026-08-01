# Quantum Tabla

> **Quantum randomness. Classical rhythm.** A real-time procedural tabla
> synthesizer driven by selectable quantum-circuit or classical
> randomness.

## Overview

**Quantum Tabla** is an interactive Python application that combines
procedural tabla synthesis with configurable randomness sources.

Choose between a **Quantum** mode using a Qiskit Aer simulated
Hadamard-measurement circuit and a **Classical** mode using NumPy's
PRNG. The resulting bitstream controls synthesis parameters and
generative rhythmic decisions in real time.

> Quantum mode uses a **quantum circuit simulator**, not physical
> quantum hardware.

## Features

-   Quantum and classical randomness modes
-   Qiskit Aer Hadamard-measurement circuit
-   Fully procedural tabla synthesis --- no prerecorded samples
-   Bayan and dayan DSP synthesis
-   Generative rhythm, accents, rests, ghost strokes, and micro-timing
-   Real-time audio playback
-   Live waveform and performance telemetry
-   Streamlit dashboard
-   Automated test suite with pytest

## Project Structure

``` text
Quantum-Tabla/
├── app.py
├── assets/
│   └── style.css
├── src/
│   ├── core/           # Randomness, configuration and instrument state
│   ├── dashboard/      # Streamlit interface
│   ├── performance/    # Real-time rhythm and audio engine
│   ├── synthesis/      # Procedural tabla DSP
│   └── utils/
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Installation

``` bash
git clone https://github.com/YKMadhav/Quantum-Tabla.git
cd Quantum-Tabla

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

``` bash
streamlit run app.py
```

Select **Classical** or **Quantum** as the randomness source and press
**Start** to begin the continuous tabla performance.

## Testing

``` bash
pip install -r requirements-dev.txt
python -m pytest
```

## Tech Stack

-   **Python**
-   **Streamlit** --- dashboard
-   **NumPy** --- numerical processing and classical randomness
-   **SciPy** --- DSP and filtering
-   **Qiskit + Qiskit Aer** --- simulated quantum circuit
-   **sounddevice** --- real-time audio playback
-   **Custom DSP** --- procedural tabla synthesis and generative
    performance

## Author

**Khatwang Madhav Yippili**

B.S. (Hons.) in Mathematical Sciences and Computing

Sri Sathya Sai Institute of Higher Learning
