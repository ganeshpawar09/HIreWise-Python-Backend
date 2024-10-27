import sys
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from warnings import filterwarnings

filterwarnings('ignore')

class LeetcodeScraper:
    def __init__(self):
        self.base_url = 'https://leetcode.com/graphql'

    def scrape_user_profile(self, username):
        output = {}

        def scrape_single_operation(operation):
            json_data = {
                'query': operation_query_dict.get(operation, ''),
                'variables': {'username': username},
                'operationName': operation
            }

            if operation == 'recentAcSubmissions':
                json_data['variables']['limit'] = 15

            try:
                response = requests.post(self.base_url, json=json_data, stream=True, verify=False)
                response.raise_for_status()  # Raise an exception for HTTP errors
                response_data = response.json()
                output[operation] = response_data.get('data', {})
            except requests.exceptions.RequestException as e:
                print(f'username: {username}', f'operation: {operation}', f'error: {e}', sep='\n')

        operation_query_dict = {
    'languageStats': '\n    query languageStats($username: String!) {\n  matchedUser(username: $username) {\n    languageProblemCount {\n      languageName\n      problemsSolved\n    }\n   }\n }\n    ',
    'skillStats': '\n    query skillStats($username: String!) {\n  matchedUser(username: $username) {\n    tagProblemCounts {\n      advanced {\n        tagName\n         problemsSolved\n      }\n      }\n  }\n}\n    ',
    'userProblemsSolved': '\n    query userProblemsSolved($username: String!) {\n    matchedUser(username: $username) {\n       submitStatsGlobal {\n      acSubmissionNum {\n        difficulty\n        count\n      }\n    }\n  }\n}\n    ',
    "userProfileCalendar": "\n    query userProfileCalendar($username: String!, $year: Int) {\n  matchedUser(username: $username) {\n    userCalendar(year: $year) {\n        streak\n      totalActiveDays\n    }\n  }\n}\n    ",
    "userContestRankingInfo": "\n    query userContestRankingInfo($username: String!) {\n  userContestRankingHistory(username: $username) {\n    rating\n  }\n}\n    "
}

        with ThreadPoolExecutor(max_workers=len(operation_query_dict)) as executor:
            executor.map(scrape_single_operation, operation_query_dict)

        if 'userContestRankingInfo' in output:
            ratings = output['userContestRankingInfo'].get('userContestRankingHistory', [])
            output['userContestRankingInfo']['userContestRankingHistory'] = ratings[-10:] 
        return output

    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scraper.py <username>")
        sys.exit(1)

    username = sys.argv[1]
    scraper = LeetcodeScraper()
    result = scraper.scrape_user_profile(username)
    print(json.dumps(result))
