import re
from io import BytesIO
from typing import Tuple, List
from functools import lru_cache
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import os

def read_chapter_content(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

def calculate_accuracy(text_content: str, questions: list) -> float:
    try:
        total_words = len(text_content.split())
        relevant_count = 0
        
        for q in questions:
            question_words = q['question'].lower().split()
            for word in question_words:
                if len(word) > 3 and word in text_content.lower():
                    relevant_count += 1
        
        accuracy = min((relevant_count / (len(questions) * 2)) * 100, 100)
        return round(accuracy, 2)
    except Exception as e:
        print(f"Error calculating accuracy: {str(e)}")
        return 0.0

def generate_quiz_questions(text_content: str, client: OpenAI, model: str):
    system_prompt = """Generate 5 thought-provoking multiple choice questions that enhance students' cognitive abilities and IQ. Include questions that:
    1. Test logical reasoning and pattern recognition
    2. Require application of concepts in novel situations
    3. Involve analysis and problem-solving
    4. Encourage creative thinking and innovation
    5. Integrate multiple concepts and ideas
    6. Challenge students to think beyond memorization
    
    Format each question as:
    Question N: [Analytical question that improves cognitive skills]
    A) [Option]
    B) [Option]
    C) [Option]
    D) [Option]
    Answer: [Letter]. [Full correct answer]
    Explanation: [Detailed explanation linking concepts and reasoning]"""

    query = f"Content:\n{text_content}\n\nCreate questions that enhance logical thinking and problem-solving abilities using the specified format."


    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        response = completion.choices[0].message.content
        return parse_quiz_response(response)
    except Exception as e:
        print(f"Error in question generation: {str(e)}")
        return None

def parse_quiz_response(response: str) -> dict:
    questions = []
    current_question = None
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith(('Question', 'Q')):
            if current_question is not None:
                questions.append(current_question)
            
            current_question = {
                'question': line.split(':', 1)[1].strip() if ':' in line else line,
                'options': [],
                'answer': '',
                'full_answer': '',
                'explanation': ''
            }
            
        elif any(line.startswith(f"{letter})") for letter in ['A', 'B', 'C', 'D']):
            option_letter = line[0]
            option_text = line[2:].strip()
            current_question['options'].append(option_text)
            current_question[f'option_{option_letter}'] = option_text

        elif line.startswith('Answer:'):
            answer_text = line.split(':', 1)[1].strip()
            # Extract letter and full answer
            match = re.match(r'([A-D])[.).]\s*(.*)', answer_text)
            if match:
                letter, full_answer = match.groups()
                current_question['answer'] = letter
                current_question['full_answer'] = full_answer
            else:
                current_question['answer'] = answer_text[0] if answer_text else ''
                current_question['full_answer'] = answer_text

        elif line.startswith('Explanation:'):
            current_question['explanation'] = line.split(':', 1)[1].strip()

    if current_question is not None:
        questions.append(current_question)

    # Validate and clean up questions
    for q in questions:
        if not q.get('answer') or q['answer'] not in 'ABCD':
            q['answer'] = 'Not specified'
            q['full_answer'] = 'Answer format error'
        if not q.get('explanation'):
            q['explanation'] = 'No explanation provided'
        if len(q.get('options', [])) != 4:
            q['options'] = ['Option A', 'Option B', 'Option C', 'Option D']

    return {'questions': questions, 'confidence': 0.9}

def generate_quiz():
    print("\nQuiz Generator")
    print("-" * 50)
    
    # Input validation
    while True:
        standard = input("Enter standard (e.g., 6): ").strip()
        if standard.isdigit():
            break
        print("Please enter a valid standard number")

    while True:
        subject = input("Enter subject (e.g., Maths): ").strip()
        if subject:
            break
        print("Please enter a valid subject")

    while True:
        chapter = input("Enter chapter number (e.g., 1): ").strip()
        if chapter.isdigit():
            break
        print("Please enter a valid chapter number")

    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("Error: OpenAI API key not found in .env file")
        return
        
    client = OpenAI(api_key=api_key)
    file_path = f"D:/bck/schoolbooks/{standard}th/{subject}/Chapter {chapter}.txt"
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    try:
        chapter_content = read_chapter_content(file_path)
        if not chapter_content:
            print("Error: Could not read chapter content")
            return

        print(f"\nGenerating questions for {subject} Chapter {chapter} (Standard {standard})")
        
        result = generate_quiz_questions(
            client=client, 
            model="gpt-4-turbo-preview",
            text_content=chapter_content
        )

        if not result:
            print("Error: Failed to generate questions")
            return

        # Calculate accuracy
        accuracy = calculate_accuracy(chapter_content, result['questions'])
        print(f"\nQuestion Generation Accuracy: {accuracy}%")
        print("=" * 50)

        print("\nGenerated Quiz Questions:")
        print("=" * 50)
        for i, q in enumerate(result['questions'], 1):
            print(f"\nQuestion {i}: {q['question']}")
            print("\nOptions:")
            for opt, option in zip(['A', 'B', 'C', 'D'], q['options']):
                print(f"{opt}) {option}")
            print(f"\nCorrect Answer: {q['answer']}. {q['full_answer']}")
            print(f"Explanation: {q['explanation']}")
            print("-" * 50)

        print("\nQuiz generation completed!")

    except Exception as e:
        print(f"Error generating quiz: {str(e)}")

if _name_ == "_main_":
    try:
        generate_quiz()
    except KeyboardInterrupt:
        print("\nQuiz generation cancelled by user")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")