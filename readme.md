# LipChat

LipChat is a lipreading application that decodes text from the movement of a speaker's mouth. It is a remake of the LipNet model, designed for end-to-end sentence-level lipreading.

![Image](/UI.png)



## Features

- Converts a sequence of video frames to text
- Uses spatiotemporal convolutions, a recurrent network, and connectionist temporal classification loss
- Trained entirely end-to-end
- Achieves high accuracy in converting speech from videos to text(95.2%)

## Getting Started

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/abuzar0013/First-Minor-Project.git
   ```
2. First, create a Conda environment to easily manage all the required libraries.

   ```sh
   conda create --name LipChat 
   ```
   Activate LipChat environment
   ```sh
   conda activate LipChat
   ```


2. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

### Usage

To run this app, please adjust all the file paths according to your system.
1. Run the LipChat app:

   ```sh
   python streamlitpp.py
   ```
   OR
   ```sh
   streamlit run /FIRST-MINOR-PROJECT/app/streamlitapp.py
   ```


2. Follow the on-screen instructions to input video files for lipreading.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Wand et al., 2016; Chung & Zisserman, 2016a for pioneering work in end-to-end lipreading
- Gergen et al., 2016 for their word-level state-of-the-art accuracy
- Easton & Basala, 1982 for studies on human lipreading performance

---
