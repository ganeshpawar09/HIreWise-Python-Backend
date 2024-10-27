from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from concurrent.futures import ThreadPoolExecutor
from warnings import filterwarnings
from typing import Dict, Any
from pydantic import BaseModel

filterwarnings('ignore')

app = FastAPI(title="LeetCode Scraper API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LeetcodeScraper:
    def __init__(self):
        self.base_url = 'https://leetcode.com/graphql'
        
    def scrape_user_profile(self, username: str) -> Dict[str, Any]:
        output = {}
        
        def scrape_single_operation(operation: str) -> None:
            json_data = {
                'query': operation_query_dict.get(operation, ''),
                'variables': {'username': username},
                'operationName': operation
            }
            
            try:
                response = requests.post(self.base_url, json=json_data, verify=False)
                response.raise_for_status()
                response_data = response.json()
                output[operation] = response_data.get('data', {})
            except requests.exceptions.RequestException as e:
                print(f'Error for {username} in {operation}: {str(e)}')
                
        operation_query_dict = {
            'languageStats': '''
                query languageStats($username: String!) {
                    matchedUser(username: $username) {
                        languageProblemCount {
                            languageName
                            problemsSolved
                        }
                    }
                }
            ''',
            'skillStats': '''
                query skillStats($username: String!) {
                    matchedUser(username: $username) {
                        tagProblemCounts {
                            advanced {
                                tagName
                                problemsSolved
                            }
                        }
                    }
                }
            ''',
            'userProblemsSolved': '''
                query userProblemsSolved($username: String!) {
                    matchedUser(username: $username) {
                        submitStatsGlobal {
                            acSubmissionNum {
                                difficulty
                                count
                            }
                        }
                    }
                }
            ''',
            'userProfileCalendar': '''
                query userProfileCalendar($username: String!, $year: Int) {
                    matchedUser(username: $username) {
                        userCalendar(year: $year) {
                            streak
                            totalActiveDays
                        }
                    }
                }
            ''',
            'userContestRankingInfo': '''
                query userContestRankingInfo($username: String!) {
                    userContestRankingHistory(username: $username) {
                        rating
                    }
                }
            '''
        }
        
        with ThreadPoolExecutor(max_workers=len(operation_query_dict)) as executor:
            executor.map(scrape_single_operation, operation_query_dict)
            
        if 'userContestRankingInfo' in output:
            ratings = output['userContestRankingInfo'].get('userContestRankingHistory', [])
            output['userContestRankingInfo']['userContestRankingHistory'] = ratings[-10:]
            
        return output

class Username(BaseModel):
    username: str

@app.post("/api/leetcode/profile")
async def get_leetcode_profile(data: Username):
    try:
        scraper = LeetcodeScraper()
        result = scraper.scrape_user_profile(data.username)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
