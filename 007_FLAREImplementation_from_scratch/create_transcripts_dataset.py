from langchain.document_loaders import YoutubeLoader
import requests
import re
from tqdm import tqdm

# some webscraping using requests and regex to get all IDs for my youtube videos
my_url = "https://www.youtube.com/@rrwithdeku8677/videos"
r = requests.get(my_url)
page = (r.text)
pattern = r'watch\?v=([^"]+)'
matches = re.findall(pattern, page, re.IGNORECASE)
ids = [x.split('=')[-1] for x in matches] # has all the IDs

base_url =  "https://www.youtube.com/watch?v="

# video_data=  []

# Using YoutubeLoader from Langchain to get the transcripts of the videos
for i,id in tqdm(enumerate(ids)):
    loader = YoutubeLoader.from_youtube_url(
        base_url + id, add_video_info=True
    )
    # print("got loader")
    data = loader.load()
    # print(data)
    # create a file for each video and write the transcript to it
    with open(f"data/{i}.txt", "w") as f:
        f.write(data[0].page_content)

