import pandas as pd
import numpy as np
import requests

def get_winedf_scrape_params():
    """
    Set parameters to be used for API call
    """
    # define params
    headers = {
        "User-Agent": ""
    }

    # Instantiate a dictionary of query strings
    # Defines the only needed payload
    params = {
        "order_by": "price",
        "order": "desc",
        "price_range_min": 1,
        "price_range_max": 2
    }

    url = "https://www.vivino.com/api/explore/explore?"

    # Performs an initial request and gathers the amount of total results
    r = requests.get(url, params=params, headers=headers)
    n_matches = r.json()['explore_vintage']['records_matched']
    
    return params, headers, url, n_matches

def get_wine_df(params, headers, url, n_matches):
    """
    Scrape vivino api and return wines and metadata
    """
    columns = ['id', 'name', 'num_ratings', 'avg_rating',
                  'region', 'region_id', 'winery', 'winery_id',
                  'varietal', 'price', 'currency']
    df = pd.DataFrame(columns=columns)
    idx = 0
    if n_matches < 25:
        n_matches = 25
    for i in range(int(n_matches / 25)):
        params['page'] = i + 1
        print(f'Requesting data from page: {params["page"]}')
        r = requests.get(url, params=params, headers=headers)
        matches = r.json()['explore_vintage']['matches']
        for match in matches:
            try:
                wid = match['vintage']['wine']['id']
            except:
                wid = np.nan
            try:
                name = match['vintage']['name']
            except:
                name = np.nan
            try:
                num_ratings = match['vintage']['statistics']['wine_ratings_count']
            except:
                num_ratings = np.nan
            try:
                avg_rating = match['vintage']['statistics']['wine_ratings_average']
            except:
                avg_rating = np.nan
            try:
                region = match['vintage']['wine']['region']['name']
            except:
                region = np.nan
            try:
                region_id = match['vintage']['wine']['region']['id']
            except:
                region_id = np.nan
            try:
                winery = match['vintage']['wine']['winery']['name']
            except:
                winery = np.nan
            try:
                winery_id = match['vintage']['wine']['winery']['id']
            except:
                winery_id = np.nan
            try:
                varietal = match['vintage']['wine']['style']['varietal_name']
            except:
                varietal = np.nan
            try:
                price = match['prices'][0]['amount']
            except:
                price = np.nan
            try:
                currency = match['prices'][0]['currency']['code']
            except:
                currency = np.nan
            df.loc[idx] = [wid, name, num_ratings, avg_rating, region, region_id,
                           winery, winery_id, varietal, price, currency] 
            print("{} done".format(idx))
            idx += 1
    return df

def get_wine_metadata_complete():
    """
    Scrape data from vivino in small clips, between low and high
    - Using smaller intervals yielded much less missing data when calling API
    - Default set here is iterating through 50 cent price ranges
    """
    #### Set up Function in case does not work ###
    low = 0
    high = 0.5
    headers = {
        "User-Agent": ""
    }
    url = "https://www.vivino.com/api/explore/explore?"
    while low <= 80:
        params = {
        "order_by": "price",
        "order": "desc",
        "price_range_min": low,
        "price_range_max": high
        }
        r = requests.get(url, params=params, headers=headers)
        n_matches = r.json()['explore_vintage']['records_matched']
        df = get_wine_df(params, headers, url, n_matches)
        if low == 0:
            df_final = df
        else:
            df_final = pd.concat([df_final, df])
        low += 0.5
        high += 0.5
        print("RANGE LOW:{} HIGH:{} COMPLETE".format(low, high))
    df_final = df_final.reset_index().drop(columns=['index'])
    df_final = df_final.drop_duplicates()
    
    return df_final

def get_reviews_df(wine_ids, headers):
    """
    Given a list of vivino wine ids, return a df with up to 500 reviews for each listed id
    """
    columns = ['review_id', 'wine_id', 'rating', 'review_text',
               'language', 'date', 'user_id']
    df = pd.DataFrame(columns=columns)
    idx = 0
    no_id = []
    for wid in wine_ids:
        page_counter = 1
        while page_counter <= 10:
            print(f'Requesting reviews from wine: {wid} and page: {page_counter}')
            # Performs the request and saves the reviews
            try:
                r = requests.get(f'https://www.vivino.com/api/wines/{wid}/reviews?per_page=50&page={page_counter}',
                                 headers=headers)
                reviews = r.json()['reviews']
                print(f'Number of reviews: {len(reviews)}') 
                if len(reviews) == 0:
                    # Breaks the loop
                    break
                # Otherwise, increments the counter
                page_counter += 1
                for review in reviews:
                    # review metedata
                    try:
                        review_id = review['id']
                    except:
                        review_id = np.nan
                    try:
                        rating = review['rating']
                    except:
                        rating = np.nan
                    try:
                        text = review['note']
                    except:
                        text = np.nan
                    try:
                        language = review['user']['language']
                    except:
                        language = np.nan
                    try:
                        date = review['created_at']
                    except:
                        date = np.nan
                    try:
                        user_id = review['user']['id']
                    except:
                        user_id = np.nan
                    df.loc[idx] = [review_id, wid, rating, text, language, date, user_id] 
                    idx += 1
            except:
                "could not retrieve reviews for {}".format(wid)
                no_id.append(wid)
    return df, wid

def scrape_save_reviews():
    """
    Save reviews in chunks to mitigate various API issues.
    """
    trigger = len(ids)
    start = 0
    end = 22010
    no_id_total = []
    while start <= 23000:
        sub = ids[start:end]
        no_id = []
        rvs, no_id = get_reviews_df(sub)
        no_id_total.append(no_id)
    #    rvs.to_csv("data/reviews_{}".format(end))
        start += 2500
        end += 2500

def count_reviews():
    """
    This function counts the total amount of reviews across the 19 csv files.
    """
    n = 2500
    review_count = 0
    while n <= 50000:
        reviews = pd.read_csv("data/reviews_{}.csv".format(n), index_col = 0)
        r_id = list(set(reviews.review_id))
        print(len(r_id))
        print()
        review_count += len(r_id)
        n += 2500
    return review_count

