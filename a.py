import json
import google.generativeai as genai

# Initialize Gemini AI client
genai.configure(api_key="AY")

def analyze_response(question: str, transcript: str) -> dict:
    """
    Sends the transcript and question to Gemini AI for grammar correction,
    enhancement, and filler word detection.
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        You are an advanced AI language model evaluating a candidate’s response to an interview question.  
        Your task is to analyze the response for **grammar correctness, fluency, conciseness, and filler word usage**.  

        ### **Instructions:**  
        - **Identify grammar mistakes**, categorizing them into specific types.  
        - **For each mistake, provide the incorrect version, the corrected version, and the type of mistake.**  
        - **Detect filler words**, listing them along with their occurrence count.  
        - **Provide an improved, more concise, and impactful version of the response.**  

        ### **Mistake Breakdown (In-Depth Categories):**  
        1. **Tense Errors:** Incorrect verb tenses (e.g., *"I go to college last year"* → *"I went to college last year"*).  
        2. **Subject-Verb Agreement:** Errors in subject-verb agreement (e.g., *"She go to work early"* → *"She goes to work early"*).  
        3. **Pronoun Usage:** Incorrect pronoun choice or ambiguity (e.g., *"Me and him went to the store"* → *"He and I went to the store"*).  
        4. **Preposition Errors:** Incorrect preposition usage (e.g., *"interested for"* → *"interested in"*).  
        5. **Word Order (Syntax):** Improper sentence structure (e.g., *"Yesterday went I to the mall"* → *"Yesterday, I went to the mall"*).  
        6. **Word Choice:** Use of incorrect words or awkward phrasing (e.g., *"Do the needful"* → *"Please take the necessary action"*).  
        7. **Redundancy & Wordiness:** Unnecessary repetition or overly complex sentences (e.g., *"I personally think that in my opinion, I believe"* → *"I think"*).  
        8. **Passive Voice Overuse:** Excessive use of passive voice (e.g., *"The project was completed by me"* → *"I completed the project"*).  

        ### **Output Format (JSON):**  
        {{
          "grammar_accuracy": "XX%",  # Percentage of grammatically correct sentences.
          "mistake_breakdown": [
            {{
              "incorrect": "Wrong sentence or phrase",
              "correct": "Corrected sentence or phrase",
              "type": "Mistake category (e.g., tense error, subject-verb agreement)"
            }},
            {{
              "incorrect": "Another wrong sentence",
              "correct": "Another corrected sentence",
              "type": "Another mistake category"
            }}
          ],
          "filler_words": {{
            "uh": X,  
            "um": Y,  
            "like": Z,  
            "you know": W  
          }},  # List of detected filler words with occurrence count.
          "enhanced_response": "A refined, more effective version of the response"
        }}

        ### **Input Data:**  
        - **Interview Question:** "{question}"  
        - **Candidate Response:** "{transcript}"  

        Analyze the response and return structured JSON output.
        """


        response = model.generate_content(prompt)
        
        # Extract and clean the response
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()

        # Try parsing as JSON
        try:
            parsed_response = json.loads(cleaned_text)
            print( {
                "grammar_accuracy": parsed_response.get("grammar_accuracy", "N/A"),
                "mistake_breakdown": parsed_response.get("mistake_breakdown", {}),
                "filler_words": parsed_response.get("filler_words", []),
                "enhanced_response": parsed_response.get("enhanced_response", "").strip()
            })
        except json.JSONDecodeError:
            return {
                "grammar_accuracy": "N/A",
                "mistake_breakdown": {},
                "filler_words": [],
                "enhanced_response": cleaned_text 
            }
    
    except Exception as e:
        return {
            "grammar_accuracy": "N/A",
            "mistake_breakdown": {},
            "filler_words": [],
            "enhanced_response": "Could not generate improved response."
        }





print(analyze_response("Tell me about yourself.","hello my name is Ganesh uchit Pawar currently working as a internet speed and also pursing my bachelor's degree from Sinhagad College of Engineering Pune where I maintain the CGPA of 8.35 and I have strong foundation in programming languages like C C plus plus Java Dart and JavaScript and good understanding of key fundamentals like data structures and algorithmic oriented programming computer networks operating system and database management system over past few years I am working on my problem solving skills I sold over 500 plus questions on lead code against some hands on experience in mobile application development and backend development I did my internship at card as a flutter developer it's a one month internship Where are replicate their entire web based system into mobile application also work on my own projects like this flatmates higher higher wise it's my last year project community where I manage multiple take related events and I am actively participating in hackathon I recently won Rockwell Automation hacked on I like to play chess and I have rating of 1200 and 1200 on the online platform chess.com that's all from my side"))
