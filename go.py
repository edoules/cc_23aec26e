"""
cc_23aec26e/go.py

Example Usage (in go.sh):
python3 go.py listings.txt product.txt > results.txt

General Strategy -- Aggressively Classify then Aggressively Delete Outliers.
STAGE 1: Classify using product.product_name in listing.title -- only accept when score is 1.0 --
STAGE 2: Disqualify based on cluster abberations -- remember, we favour specificity over sensitivity here --
         - consider each product_name to be a cluster
         - listings contain additional fields: family, manufacturer, model, (currency, price)
         - (Skipped This Step) delete items in each cluster that appear to be outliers
         - but DO ensure manufacturers match between listings and products
         - be aggressive in deletion: we care about specificity over sensitivity.
STAGE 3: Print results as per spec --
"""

def jsonl_handle_to_dataframe(handle):
    """Return a dataframe from a file handle containing jsonl documents."""
    import json
    import pandas as pd
    from collections import defaultdict
    
    acc = []
    for line in handle:
        linedict = defaultdict(lambda:None, json.loads(line)) # allows missing keys.
        acc += [linedict]
    
    key_collect = set() # in case new columns are added in later jsonl lines.
    for row in acc:
        key_collect |= set(row.keys())

    colmajor = {k:[] for k in key_collect}
    for row in acc:
        for k in key_collect:
            colmajor[k] += [row[k]]

    df = pd.DataFrame(colmajor)
    
    return df

def has_perfect_alphanumeric_subsequence(query_string, reference_string):
    """Return 1. if a reference string alphanumeric subsequence from product (reference) is found in listings (query) else 0.;"""
    # - initial cheapest fastest test before other tests

    import re

    if type(query_string) == type(None) or type(reference_string) == type(None):
        return 0.

    query_string = query_string.lower()
    query_string = re.sub(r'([^\w]|_)+', '', query_string, re.UNICODE)

    reference_string = reference_string.lower()
    reference_string = re.sub(r'([^\w]|_)+', '', reference_string, re.UNICODE)

    if reference_string in query_string:
        return 1.

    return 0.

def string_similarity_score(query_string, reference_string):
    """Return a percentage match between strings of tokens."""
    # - limitation: word order is ignored
    # - limitation: words in one string can match multiple words in another string (and vice versa)
    # - can't use jaccard index as sets here should be treated assymetrically
    # - query_string (listings)
    # - reference_string (products) treat as denominator, tho paradoxically is usually shorter than query_string
    # - jaccard is |q & a| / |q | a|
    # - assymetry means use |q & a| / |a| -- allows q to get arbitrarily long
    # - finally, we want inexact matches for tokens in q and tokens in a; so take sum over pairwise scores --
    # - pairwise scores defined as token_similarity_score() - in future, use argmax to have single pairs match
    
    from itertools import product
    import re

    if type(query_string) == type(None) or type(reference_string) == type(None):
        return 0.

    query_token_set = re.sub(r'([^\w]|_)+', ' ', query_string, re.UNICODE)
    query_token_set = query_token_set.lower().split()
    query_token_set = set(query_token_set)

    reference_token_set = re.sub(r'([^\w]|_)+', ' ', reference_string, re.UNICODE)
    reference_token_set = reference_token_set.lower().split()
    reference_token_set = set(reference_token_set)

    numer = len(query_token_set & reference_token_set)

    denom = len(reference_token_set)
    score = 1. * numer / denom

    return score

def string_similarity_score_allow_incorrect_spacing(query_string, reference_string):
    """Return percentage match between strings of tokens, allowing adjacent pairs of tokens to be glued together."""
    # - treats special cases of names like "Blahmark 300HS" vs "Blahmark 300 HS"
    # - limitation: considers adjacent pairs one-at-a-time, instead of all possible adjacencies simultaneously
    # - descends from string_similarity_score()
    # - still use modified normalized set intersection again |q & a| / |a|
    
    from itertools import product
    import re

    if type(query_string) == type(None) or type(reference_string) == type(None):
        return 0.

    best_score = 0.

    query_token_list = re.sub(r'([^\w]|_)+', ' ', query_string, re.UNICODE)
    query_token_list = query_token_list.lower().split()

    for itoken in range(0, len(query_token_list) -1):

        jtoken = itoken + 1
        curr_glued_adjacent = query_token_list[itoken] + query_token_list[jtoken]
        non_adjacent_tokens = [token for ktoken, token in enumerate(query_token_list) if ktoken not in (itoken, jtoken)]
        query_token_set = [curr_glued_adjacent] + non_adjacent_tokens

        query_token_set = set(query_token_set)

        reference_token_set = re.sub(r'([^\w]|_)+', ' ', reference_string, re.UNICODE)
        reference_token_set = reference_token_set.lower().split()
        reference_token_set = set(reference_token_set)

        numer = len(query_token_set & reference_token_set)

        denom = len(reference_token_set)
        score = 1. * numer / denom

        best_score = max(best_score, score)

    return best_score

def get_listings_best_column_match_and_score(products_column, listings_column):
    """Find the best-matching product given listings -- comparing columns to columns; returns two dataframe columns."""
    # - limitation: rows scoring identically will have tie-broken reverse-alphabetically

    from sys import stderr

    print("STAGE 1: -- Aggressively classify based on listings.title to product.product_name matches --", file=stderr)

    products_column = products_column
    listings_column = listings_column

    best_match_list = []
    best_score_list = []

    cache = {} # memoize ... evaluating these scores takes a while -- remember strings we've already seen.

    count_matched = 0
    count_cache_hits = 0

    for irow, l_man in enumerate(listings_column):

        if irow and int(irow % (len(listings_column)/1000)) == 0:
            percent_done = 100. * irow / len(listings_column)
            print("Progress -- %04.1f%% items:%d -- matched:%2d%% cache_hits:%2d%%" % (
                    percent_done,
                    irow,
                    int(100. * count_matched / irow),
                    int(100. * count_cache_hits / irow),
                ),
                file=stderr
            )

        if l_man in cache:
            count_cache_hits += 1
            best_score, best_product = cache[l_man]
            best_match_list += [best_product]
            best_score_list += [best_score]
            if best_score == 1: count_matched += 1
            continue

        score_to_match = [] # (score, product manufacturer)
        for p_man in products_column:
            
            lp_score = 0.

            if lp_score < 1.:
                lp_score = max(lp_score, has_perfect_alphanumeric_subsequence(l_man, p_man))
            
            if lp_score < 1.:
                lp_score = max(lp_score, string_similarity_score(l_man, p_man))

            if lp_score < 1.:
                lp_score = max(lp_score, string_similarity_score_allow_incorrect_spacing(l_man, p_man))

            if type(p_man) == type(None): p_man = ''

            score_to_match += [(lp_score, p_man)]

        best_score, best_product = max(score_to_match)
        
        if best_score == 1: count_matched += 1

        best_match_list += [best_product]
        best_score_list += [best_score]

        cache[l_man] = best_score, best_product

    percent_done = 100. * irow / len(listings_column)
    print("Progress -- DONE items:%d -- matched:%2d%% cache_hits:%2d%%" % (
            len(listings_column),
            int(100. * count_matched / len(listings_column)),
            int(100. * count_cache_hits / len(listings_column)),
        ),
        file=stderr
    )

    return best_match_list, best_score_list

def disqualify_listings_based_on_cluster(product_df, listings_df):
    """Mark incorrectly matched listings based on some heuristics -- only heuristic implemented here is matching manufacturer."""

    from sys import stderr
    from collections import defaultdict

    print("STAGE 2: -- Aggressively reject items based on other listings.* fields in each product cluster", file=stderr)
    print("         -- only considers listings.manufacturer vs products.manufacturer", file=stderr)
    print("         -- other fields can be considered in future, but this appears to do a reasonable job", file=stderr)

    out_cluster_df = None

    for icluster, product_name in enumerate(product_df.index):

        listing_cluster_df = listings_df[listings_df["fk_product_df"] == product_name].copy()
        
        # DO STAGE 2.1. filtering out by: manufacturer name field --

        product_row = product_df[product_df.index == product_name]

        product_manufacturer = product_row["manufacturer"][0]

        listing_manufacturer_scores = []
        for manufacturer in listing_cluster_df["manufacturer"]:

            manufacturer_score = 1. if product_manufacturer == manufacturer else 0.

            if manufacturer_score < 1.:
                manufacturer_score = max(manufacturer_score, has_perfect_alphanumeric_subsequence(manufacturer, product_manufacturer))

            if manufacturer_score < 1.:
                manufacturer_score = max(manufacturer_score, string_similarity_score(manufacturer, product_manufacturer))

            if manufacturer_score < 1.:
                manufacturer_score = max(manufacturer_score, string_similarity_score_allow_incorrect_spacing(manufacturer, product_manufacturer))

            listing_manufacturer_scores += [manufacturer_score]

        listing_cluster_df["product_manufacturer"] = product_manufacturer
        listing_cluster_df["manufacturer_score"] = listing_manufacturer_scores

        # DO STAGE 2.2. [OMITTED] filtering out by: family misfit --
        # DO STAGE 2.3. [OMITTED] filtering out by: model misfit --
        # DO STAGE 2.4. [OMITTED] filtering out by: currency and price misfit --

        if type(out_cluster_df) == type(None):
            out_cluster_df = listing_cluster_df.copy()
        else:
            out_cluster_df = out_cluster_df.append(listing_cluster_df)

    return out_cluster_df

def print_results_as_jsonl(product_df, listings_df, filehandle):
    """Print resulting dataframe into jsonl documents -- exactly only columns in listings specs are included."""

    import json

    print("STAGE 3: -- Print results to stdout (go.sh pipes to file by default)", file=stderr)
    
    for icluster, product_name in enumerate(product_df.index):

        outjsonl = {
            "product_name":product_name,
            "listings":[],
        }

        listing_cluster_df = listings_df[listings_df["fk_product_df"] == product_name].copy()

        # we only care about the original jsonl keys --
        listing_cluster_df = listing_cluster_df[["title", "manufacturer", "currency", "price"]]

        for irow, row in listing_cluster_df.iterrows():
            listing_dict = dict(row)
            outjsonl["listings"] += [listing_dict]

        print(json.dumps(outjsonl, ensure_ascii=False), file=filehandle)

    return

if __name__ == "__main__":

    from sys import argv, stderr, stdout
    import os
    import pandas as pd

    pd.set_option('display.width', 160) # DEV: set pandas dataframe printout to allow 160 characters wide

    products_filename = argv[1]
    list_filename = argv[2]

    with open(products_filename) as h:
        product_df = jsonl_handle_to_dataframe(h)
        product_df = product_df.set_index(['product_name'])

    with open(list_filename) as h:
        listings_df = jsonl_handle_to_dataframe(h)

    # listings_df = listings_df[:1000] # DEV: develop using the first few listings, test on the whole set.

    # DO STAGE 1: CLASSIFY
    title_name_match, title_name_score = get_listings_best_column_match_and_score(
        product_df.index, listings_df["title"]
    )
    listings_df["fk_product_df"] = title_name_match
    listings_df["product_classify_score"] = title_name_score

    # Aggressively Drop Non-Matches ...
    listings_df = listings_df[listings_df["product_classify_score"] == 1.] # aggressively drop non-matches.
    listings_df = listings_df.reset_index(drop=True)
    del listings_df["product_classify_score"] # we're done with this now.

    # DO STAGE 2: DELETE DISQUALIFIED MATCHES
    listings_df = disqualify_listings_based_on_cluster(product_df, listings_df)
    listings_df = listings_df[listings_df["manufacturer_score"] == 1.]
    listings_df = listings_df.reset_index(drop=True)
    del listings_df["manufacturer_score"] # done with this.

    # DO STAGE 3: PRINT RESULTS
    print_results_as_jsonl(product_df, listings_df, stdout)
