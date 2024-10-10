import random
import matplotlib.pyplot as plt

def calculate_tour_distance(tour, distances):
    """
    Calculate the total distance of a given tour by summing the distances between consecutive cities.
    Additionally, account for the return distance to the starting city.

    :param tour: A list representing the order of cities in the tour.
    :param distances: A 2D matrix of distances between cities.
    :return: The total distance of the tour.
    """
    distance = 0
    num_cities = len(tour)
    for i in range(num_cities - 1):
        from_city, to_city = tour[i], tour[i + 1]
        distance += distances[from_city][to_city]
    # Return to the starting city
    distance += distances[tour[-1]][tour[0]]
    return distance

def apply_two_opt_swap(tour, i, j):
    """
    Perform a 2-opt swap on the tour, which reverses the order of cities between indices i and j.

    :param tour: The tour to apply the 2-opt swap on.
    :param i: The start index of the swap.
    :param j: The end index of the swap.
    :return: The tour with the 2-opt swap applied.
    """
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour

def two_opt_algorithm(tour, distances):
    """
    Implement the 2-opt algorithm to improve the given tour for the Traveling Salesman Problem (TSP).

    :param tour: A list representing the initial tour.
    :param distances: A matrix of distances between cities.
    :return: The best tour found and its corresponding distance.
    """
    num_cities = len(tour)
    best_tour = tour
    best_distance = calculate_tour_distance(tour, distances)
    improved = True

    while improved:
        improved = False
        for i in range(1, num_cities - 2):
            for j in range(i + 2, num_cities):
                new_tour = apply_two_opt_swap(tour, i, j)
                new_distance = calculate_tour_distance(new_tour, distances)

                if new_distance < best_distance:
                    best_tour = new_tour
                    best_distance = new_distance
                    improved = True
                    tour = new_tour

    return best_tour, best_distance

def plot_tour(tour, cities):
    x = [city[0] for city in cities]
    y = [city[1] for city in cities]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, marker='o', color='b', label='Cities')
    
    # Draw lines between cities to represent the tour
    for i in range(len(tour) - 1):
        city1 = cities[tour[i]]
        city2 = cities[tour[i + 1]]
        plt.plot([city1[0], city2[0]], [city1[1], city2[1]], 'r')

    # Return to the starting city
    city1 = cities[tour[-1]]
    city2 = cities[tour[0]]
    plt.plot([city1[0], city2[0]], [city1[1], city2[1]], 'r')

    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('TSP Solution')
    plt.legend()
    plt.grid()
    plt.show()

# Example usage:
if __name__ == '__main__':
    # Define the cities and their coordinates
    # random.seed(42)
    num_cities = 100
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]

    
    # Calculate distances between cities
    num_cities = len(cities)
    distances = [[0] * num_cities for _ in range(num_cities)]
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            x1, y1 = cities[i]
            x2, y2 = cities[j]
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            distances[i][j] = distances[j][i] = dist

    # Create an initial tour (e.g., starting from city 0)
    initial_tour = list(range(num_cities))
    
    random.shuffle(initial_tour)  # Shuffle the initial tour

    # Optimize the tour using 2-opt
    optimized_tour, optimized_distance = two_opt_algorithm(initial_tour, distances)

    # Plot the TSP solution
    plot_tour(optimized_tour, cities)
    # Print the results
    print("Initial Tour:", initial_tour)
    print("Optimized Tour:", optimized_tour)
    print("Tour Distance:", optimized_distance)
