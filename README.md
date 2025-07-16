# Web-Based AI Study Guide and Quiz App

This project is an interactive AI Study Guide built using HTML, CSS, and JavaScript. It helps users learn key topics in Artificial Intelligence and test their understanding through a basic keyword-based self-evaluation quiz.

---

## Features

- Topic-based content delivery (Concepts, History, Trends, etc.)
- Question-wise learning under each topic
- "Wanna have test?" button to enable test mode
- Keyword-based evaluation of written answers
- Simple, responsive UI using CSS

---

## Technologies Used

- HTML5
- CSS3
- Vanilla JavaScript (no frameworks)

---

## How It Works

1. **Topic & Question Selection**  
   Users select a topic from the dropdown and then choose a related question. The corresponding answer appears below.

2. **Self-Test Mode**  
   Users can write their answer and receive feedback based on keyword matching logic.

3. **Keyword-Based Scoring**  
   The score is calculated by comparing the user's input with expected keywords related to the selected question.

---

## How to Run Locally

1. Download or clone this repository.
2. Ensure the following file structure:

├── index.html
├── style.css
├── script.js
└── images/
└── background.jpg

3. Open `index.html` in any modern web browser (e.g., Chrome, Firefox).

---

## Folder Structure
/
├── index.html # Main HTML file
├── style.css # Styles for UI
├── script.js # JS logic (data, interactivity, scoring)
└── images/
└── background.jpg # Optional background used in styling


---

## Future Enhancements

- Add more topics and sub-questions
- Use natural language processing (NLP) for more accurate answer checking
- Save user progress and scores locally
- Add timer-based quizzes with marks

# Question Bank Application

A web-based application for generating, managing, and printing questions for academic subjects in Computer Science.

---

## Features

- **Question Generation**  
  Automatically generates random questions from predefined question banks.

- **Subject Coverage**  
  Includes three core subjects:
  - Cryptography and Network Security
  - Computer Science and Applications
  - Digital Image Processing

- **Answer Keys**  
  Provides answer keys for all subjects (Parts A, B, and C).

- **Custom Questions**  
  Allows users to create and print their own questions easily.

- **Responsive Design**  
  Works on both desktop and mobile browsers.

- **Print Functionality**  
  Enables users to print generated or custom-created questions with clean formatting.

---

## Technologies Used

- HTML5  
- CSS3  
- JavaScript

---

## File Structure

/
├── index.html # Main application entry point
├── styles.css # Global styles
├── script.js # Main application logic
├── generate.html # Question generation page
├── makequestion.html # Custom question creation page
├── makequestion.js # Printing functionality for custom questions
├── subjects.html # Subject listing page
├── answerkey.html # Answer key page
└── logout.html # Logout confirmation page


---

## How to Use

### 1. Main Menu (`index.html`)
Access all features of the application via the main menu buttons.

### 2. Generate Questions (`generate.html`)
- Enter your request using this format:  
  **`[Subject] Part [A/B/C] generate [number] questions`**  
  _Example:_ `Computer Science Part A generate 5 questions`

### 3. Create Custom Questions (`makequestion.html`)
- Enter your custom question in the text area.
- Click **Print** to generate a printable view.

### 4. View Answer Keys (`answerkey.html`)
- Select any subject to view its full answer key for all parts.

### 5. View Subjects (`subjects.html`)
- Browse the list of available subjects along with staff details and registration numbers.

---

## Installation

No installation required.  
Simply open `index.html` in any modern web browser to start using the application.

---

## Browser Compatibility

The application is compatible with the following browsers:
- Google Chrome (latest version)
- Mozilla Firefox (latest version)
- Microsoft Edge (latest version)
- Safari (latest version)

---

## License

This project is open-source and available for educational purposes only.  
Feel free to modify and use with appropriate attribution.


---

## Future Enhancements

- User authentication for login/logout features
- Add difficulty levels for questions
- Expand subject list across multiple domains
- Enable question tagging and categorization

---

## Contributing

Contributions are welcome!  
If you'd like to contribute:
1. Fork the repository.
2. Make your changes.
3. Submit a pull request.

---

## Author

**Zafar Farhaan Z.H**  
Email: zfarhaan68@gmail.com  
LinkedIn: https://in.linkedin.com/in/zafar-farhaan-a64756355


