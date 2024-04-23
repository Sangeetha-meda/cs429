from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Documents to be indexed
documents = [
    "While your online purchases are on the road, Laura Lane is keeping on an eye on the impact of the journey. As UPS's chief corporate affairs and sustainability officer, Lane is working to trim the emissions of the logistics firm, which is responsible for delivering 22 million packages every day across more than 200 countries and territories.",
    "Most of these deliveries are driven on a fleet of 125,000 package cars, vans, lorries and motorcycles, or flown on a fleet of around 500 leased, owned and chartered aircraft. The combined emissions from all these vehicles add up. According to UPS data seen by the BBC, its air and ground operations produced a total of 14 million tonnes of CO2 or equivalent emissions in 2023."
    "Lane isn't content to simply wait for the perfect solution, be that an emissions-free aircraft fuel or a fully battery-electric lorry. She's figuring out the answer as she goes – and the vast amount of data collected by UPS is critical to reaching the company's sustainability targets. UPS is an engineering company and a technology company at its foundation, Lane tells the BBC, and so we're always looking for efficiencies. And efficiencies equal sustainability",
    "So far, UPS is finding success. In 2023, it logged an 8.1% decrease in Scope 1 emissions (pollution UPS produces directly), Scope 2 emissions (pollution from sources like electricity UPS uses to power its facilities) and Scope 3 emissions (pollution associated with the company's suppliers and customers use of UPS' services). That's an improvement from 6.9% the previous year. "
]

def perform_search(query, tfidf_matrix, vectorizer, top_k=2):
    """
    Perform search using cosine similarity based on TF-IDF representations.
    """
    # Convert query to TF-IDF vector using the same vectorizer
    query_vec = vectorizer.transform([query])

    # Compute cosine similarity between query vector and document vectors
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix)

    # Get indices of top-K most similar documents
    top_indices = cosine_similarities.argsort()[0][-top_k:][::-1]

    # Get top-K results along with cosine similarity scores
    top_results = [(documents[idx], cosine_similarities[0][idx]) for idx in top_indices]
    return top_results

if __name__ == "__main__":
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Save TF-IDF matrix and vectorizer to pickle files
    with open('tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # search query
    search_query = "emissions-free aircraft fuel "

    # Perform search and display results
    search_results = perform_search(search_query, tfidf_matrix, vectorizer, top_k=2)

    # Display search results
    print(f"Search Query: {search_query}\n")
    for result, score in search_results:
        print(f"Document: {result}\nCosine Similarity Score: {score}\n")
