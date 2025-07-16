
const topics = {
    concepts: [
        { question: "What is Artificial Intelligence?", answer: "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. These machines are programmed to think and act like humans and can perform tasks such as learning, problem-solving, and decision-making." },
        { question: "What are the types of AI?", answer: "There are three types of AI: Narrow AI, General AI, and Super AI. Narrow AI is specialized in one task, General AI is at a human level, and Super AI surpasses human intelligence." },
        { question: "What are the goals of AI?", answer: "The goals of AI include solving problems, reasoning, understanding language, and perception. AI aims to create systems that can function autonomously and make decisions." }
    ],
    history: [
        { question: "Who is considered the father of AI?", answer: "John McCarthy, who coined the term 'Artificial Intelligence' in 1956, is considered the father of AI. He organized the first conference on AI at Dartmouth College." },
        { question: "What are the significant milestones in AI?", answer: "Some milestones include the creation of the first AI program, 'Logic Theorist,' in 1955, IBM's Deep Blue defeating chess champion Garry Kasparov in 1997, and Google's AlphaGo defeating Go champion Lee Sedol in 2016." },
        { question: "When did AI start?", answer: "AI as a field began in 1956 during a conference at Dartmouth College where the term 'Artificial Intelligence' was first used." }
    ],
    trends: [
        { question: "What are the current trends in AI?", answer: "Current trends in AI include natural language processing (NLP), autonomous systems, AI ethics and fairness, AI in healthcare, and AI-driven automation." },
        { question: "How is AI being used in healthcare?", answer: "AI in healthcare is being used for predictive diagnostics, personalized treatment plans, drug discovery, and even robotic surgery. AI can process vast amounts of medical data and identify patterns to improve patient care." },
        { question: "What is AI in autonomous vehicles?", answer: "AI is a key technology in the development of autonomous vehicles. AI algorithms enable vehicles to detect their environment, make driving decisions, and safely navigate without human intervention." }
    ],
    human_vs_machine: [
        { question: "What are the differences between human and machine intelligence?", answer: "Human intelligence is biological and capable of emotions, creativity, and intuition, while machine intelligence is based on algorithms and limited to specific tasks without emotional or intuitive capabilities." },
        { question: "Can AI surpass human intelligence?", answer: "There is debate about whether AI can surpass human intelligence. While AI can outperform humans in specific tasks, such as data analysis and pattern recognition, it lacks general intelligence and emotional understanding." },
        { question: "What are the risks of AI replacing humans?", answer: "The risks include job displacement, ethical dilemmas, loss of human creativity, and dependency on machines for decision-making, leading to potential societal disruptions." }
    ],
    anna_university: [
        { question: "What is the Turing Test?", answer: "The Turing Test, developed by Alan Turing, is a test of a machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human." },
        { question: "What are Expert Systems in AI?", answer: "Expert Systems are computer programs that mimic the decision-making ability of a human expert. They use a set of rules to analyze and interpret data." },
        { question: "What is Machine Learning?", answer: "Machine Learning is a subset of AI that allows systems to automatically learn and improve from experience without being explicitly programmed." },
        { question: "Explain neural networks in AI.", answer: "Neural networks are computing systems inspired by the human brain's network of neurons. They are used to recognize patterns, classify data, and make decisions in AI applications." }
    ]
};


document.getElementById('topicSelect').addEventListener('change', function() {
    const selectedTopic = this.value;
    const subTopicDiv = document.getElementById('subTopicDiv');
    const subTopicSelect = document.getElementById('subTopicSelect');
    const answerArea = document.getElementById('answer');

 
    subTopicSelect.innerHTML = '<option value="" disabled selected>Choose a question</option>';
    answerArea.textContent = '';

  
    if (selectedTopic !== "") {
        const subTopics = topics[selectedTopic];
        subTopics.forEach(sub => {
            const option = document.createElement('option');
            option.value = sub.question;
            option.textContent = sub.question;
            subTopicSelect.appendChild(option);
        });
        subTopicDiv.style.display = 'block';
        subTopicDiv.classList.add('fade-in');
    } else {
        subTopicDiv.style.display = 'none';
    }
});


document.getElementById('subTopicSelect').addEventListener('change', function() {
    const selectedQuestion = this.value;
    const selectedTopic = document.getElementById('topicSelect').value;
    const answerArea = document.getElementById('answer');

    const selectedSubTopic = topics[selectedTopic].find(sub => sub.question === selectedQuestion);

    if (selectedSubTopic) {
        answerArea.textContent = selectedSubTopic.answer;
        document.querySelector('.answer-area').style.display = 'block';  // Show the answer area when a subtopic is selected
    }
});

const testKeywords = {
    
    "What is Artificial Intelligence?": ["intelligence", "machines", "tasks", "learning", "problem-solving", "decision-making"],
    "What are the types of AI?": ["Narrow", "General", "Super", "human", "task", "surpasses", "level"],
    "What are the goals of AI?": ["solving", "problems", "reasoning", "language", "perception", "autonomous", "decisions"],

   
    "Who is considered the father of AI?": ["John McCarthy", "father", "Artificial Intelligence", "1956", "conference", "Dartmouth College"],
    "What are the significant milestones in AI?": ["milestones", "AI", "Logic Theorist", "Deep Blue", "Kasparov", "AlphaGo", "Lee Sedol"],
    "When did AI start?": ["AI", "began", "1956", "conference", "Dartmouth College", "Artificial Intelligence"],

   
    "What are the current trends in AI?": ["current trends", "NLP", "autonomous systems", "ethics", "fairness", "healthcare", "automation"],
    "How is AI being used in healthcare?": ["AI", "healthcare", "predictive", "diagnostics", "personalized", "treatment", "drug discovery", "robotic surgery", "data", "patterns"],
    "What is AI in autonomous vehicles?": ["AI", "autonomous", "vehicles", "environment", "driving", "decisions", "navigate", "human intervention"],

   
    "What are the differences between human and machine intelligence?": ["human", "intelligence", "biological", "emotions", "creativity", "intuition", "machine", "algorithms", "tasks"],
    "Can AI surpass human intelligence?": ["AI", "surpass", "human intelligence", "tasks", "data analysis", "pattern recognition", "general intelligence", "emotional understanding"],
    "What are the risks of AI replacing humans?": ["risks", "potential societal disruptions", "leading to", "human", "ethical dilemmas", "creativity", "dependency", "decision-making"],


    "What is the Turing Test?": ["Turing Test", "Alan Turing", "test", "machine's", "intelligent", "behavior", "indistinguishable", "human"],
    "What are Expert Systems in AI?": ["Expert Systems", "computer", "programs", "mimic", "decision-making", "rules", "analyze", "interpret", "data"],
    "What is Machine Learning?": ["Machine Learning", "subset", "AI", "learn", "improve", "experience", "programmed"],
    "Explain neural networks in AI.": ["neural networks", "computing", "systems", "human brain", "neurons", "patterns", "classify", "data", "decisions"]
};


document.getElementById('takeTestBtn').addEventListener('click', function() {
   
    document.querySelector('.answer-area').style.display = 'none';

 
    document.getElementById('testSection').style.display = 'block';
    document.getElementById('testFeedback').textContent = '';
});


document.getElementById('submitTest').addEventListener('click', function() {
    const selectedQuestion = document.getElementById('subTopicSelect').value;
    const userInput = document.getElementById('testInput').value.toLowerCase();
    const feedbackDiv = document.getElementById('testFeedback');

  
    if (testKeywords[selectedQuestion]) {
        const keywords = testKeywords[selectedQuestion];
        const matches = keywords.filter(keyword => userInput.includes(keyword.toLowerCase()));
        const score = (matches.length / keywords.length) * 100;
        let feedbackMessage = `Your score: ${Math.round(score)}%. You included ${matches.length} of ${keywords.length} important keywords. `;
        if (score === 100) {
            feedbackMessage += "Awesome job! You nailed it! 💯 Keep up the fantastic work!";
        } else if (score >= 70) {
            feedbackMessage += "Great effort! You're really on the right track! 🌟 Just a little more to go!";
        } else if (score >= 50) {
            feedbackMessage += "Good try! You're getting there! ✨ Keep practicing and you'll master this topic!";
        } else {
            feedbackMessage += "Don't give up! You're learning and improving. 🙌 Keep at it, and you'll see results soon!";
        }

        feedbackDiv.textContent = feedbackMessage;
    } else {
        feedbackDiv.textContent = "This question doesn't have keywords set for evaluation.";
    }
});
