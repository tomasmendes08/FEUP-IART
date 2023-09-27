import statistics
import random
import math
import copy

name = '' #
algorithm = 0  # 1 - first accept hill climbing, 2 - steepest hill climbing, 3 - simulated annealing, 4 - genetic
evaluation = 0  # 1 - real score (takes too long), 2 - score (duplicates), 3 - heuristic
solution_type = 0  # 1 - random, 2 - greedy score, 3 - greedy sign up

alpha = 0.99
elitism = True

n_books = 0
n_libraries = 0
n_days = 0
book_scores = []
libraries = []


def sort_dict(d):
    return {k: v for k, v in reversed(sorted(d.items(), key=lambda item: item[1]))}


class Library(object):
    def __init__(self, id, n_books, signup, debit, books):
        self.id = id
        self.n_books = n_books
        self.signup = signup
        self.debit = debit
        self.books = sort_dict(books)
        self.n_best_scores = []
        self.total_score = 0
        self.scanned = []

        for book in self.books:
            self.total_score += self.books[book]
            self.n_best_scores.append(self.total_score)


class Solver(object):
    def __init__(self, file):
        self.solution = []
        self.file = file

    ### READ FROM FILE ###

    def input(self):
        global n_books, n_libraries, n_days, book_scores, libraries, evaluation

        path = 'input/' + self.file

        with open(path, 'r') as f:
            [n_books, n_libraries, n_days] = [
                int(el) for el in f.readline().split()]
            scoresLine = f.readline().split()

            for el in scoresLine:
                book_scores.append(int(el))

            for i in range(n_libraries):
                [n_books, signup, debit] = [int(el)
                                            for el in f.readline().split()]
                books = {}

                books_line = [int(el) for el in f.readline().split()]

                for book in books_line:
                    books.update({book: book_scores[book]})

                lib = Library(i, n_books,
                              signup, debit, books)
                libraries.append(lib)

    ### WRITE TO FILE ###

    def output(self):
        path = 'output/' + self.file

        ttd = n_days  # time to deadline
        ntsu = 0  # number of libraries to sign up
        scanned = []

        for i in self.solution:
            ttd -= libraries[i].signup

            if ttd <= 0:
                break

            ntsu += 1

            bts = ttd * libraries[i].debit  # books to scan

            for book in libraries[i].books:
                if bts <= 0:
                    break
                if book not in scanned:
                    scanned.append(book)
                    libraries[i].scanned.append(book)
                    bts -= 1

        with open(path, 'w') as f:
            f.write(str(ntsu) + '\n')

            for i in range(ntsu):
                f.write(str(self.solution[i]) + " " + str(len(libraries[self.solution[i]].scanned)) + '\n')

                for b in libraries[self.solution[i]].scanned:
                    f.write(str(b) + " ")

                f.write('\n')

        return 0

    ### EVALUATION FUNCTIONS ###

    def evaluate(self, solution):
        if evaluation == 1:
            return self.score(solution)  # real score, but slow
        elif evaluation == 2:
            return self.score_duplicates(solution)  # very close to real score, but fast
        elif evaluation == 3:
            return self.heuristic_score(solution)  # a different way to assess fitness, but fast

        return 0

    def heuristic_score(self, solution):
        score = 0
        ttd = 0  # time to deadline

        for i in solution:
            ttd = n_days - libraries[i].signup

            bts = ttd * libraries[i].debit  # books to scan

            if bts < len(libraries[i].books):
                score += libraries[i].n_best_scores[bts]/libraries[i].signup
            else:
                score += libraries[i].total_score/libraries[i].signup

        return score

    def score_duplicates(self, solution):
        score = 0
        ttd = n_days  # time to deadline

        for i in solution:
            ttd -= libraries[i].signup

            if ttd <= 0:
                return score

            bts = ttd * libraries[i].debit  # books to scan

            if bts < len(libraries[i].books):
                score += libraries[i].n_best_scores[bts]
            else:
                score += libraries[i].total_score

        return score

    def score(self, solution):
        score = 0
        ttd = n_days  # time to deadline
        scanned = []

        for i in solution:
            ttd -= libraries[i].signup

            if ttd <= 0:
                return score

            bts = ttd * libraries[i].debit  # books to scan

            for book in libraries[i].books:
                if bts <= 0:
                    break
                if book not in scanned:
                    score += libraries[i].books[book]
                    scanned.append(book)
                    bts -= 1

        return score

    def solve(self, alg):
        if alg == 1:
            self.solution = self.first_accept_hillclimb()
        elif alg == 2:
            self.solution = self.steepest_hillclimb()
        elif alg == 3:
            self.solution = self.annealing()
        elif alg == 4:
            self.solution = self.genetic()

        return self.score(self.solution)

    ### FIRST ACCEPT HILL CLIMBING ###

    def first_accept_hillclimb(self):
        current_solution = self.get_initial_solution()  # initial solution

        done = False

        while not done:
            list_of_neighbours = self.get_neighbours(
                current_solution)  # get all neighbours
            best_neighbour = copy.copy(current_solution)

            for x in range(len(list_of_neighbours)):
                if self.evaluate(list_of_neighbours[x]) > self.evaluate(best_neighbour):
                    # accept first neighbour with a higher score than the current solution
                    best_neighbour = list_of_neighbours[x]
                    break

            if current_solution == best_neighbour:
                done = True  # end while loop when we don't have a better solution
            else:
                current_solution = best_neighbour

        return current_solution

    ### STEEPEST HILL CLIMBING ###

    def steepest_hillclimb(self):
        current_solution = self.get_initial_solution()  # initial solution

        done = False
        while not done:
            list_of_neighbours = self.get_neighbours(
                current_solution)  # get all neighbours
            best_neighbour = copy.copy(current_solution)
            for x in range(len(list_of_neighbours)):
                if self.evaluate(list_of_neighbours[x]) > self.evaluate(best_neighbour):
                    # accept the neighbour with the highest score
                    best_neighbour = list_of_neighbours[x]

            if current_solution == best_neighbour:
                done = True  # end while loop when we don't have a better solution
            else:
                current_solution = best_neighbour

        return current_solution

    ### SIMULATED ANNEALING ###

    def annealing(self):
        current_solution = self.get_initial_solution()  # initial solution
        current_solution_score = self.evaluate(current_solution)

        best_solution = current_solution
        best_solution_score = current_solution_score
        print("Current Solution Score: " + str(best_solution_score))
        initial_temp = 100000000  # initial temperature

        while round(initial_temp) != 0:  # end while loop when temp is approximately 0
            list_of_neighbours = self.get_neighbours(current_solution)
            random_neighbour = random.choice(
                list_of_neighbours)  # choose a random neighbour
            neighbour_score = self.evaluate(random_neighbour)
            print("Neighbour Solution Score: " + str(neighbour_score))

            delta = neighbour_score - current_solution_score

            # accept neighbour if he has a higher score or accept it with a probability of e^(-delta/initial_temp)
            if delta > 0 or random.random() <= (1 / (1 + math.exp(delta/initial_temp))):
                current_solution = random_neighbour
                current_solution_score = neighbour_score

            # check if current_solution has a higher score than our best solution so far
            if current_solution_score > best_solution_score:
                best_solution = current_solution
                best_solution_score = current_solution_score

            initial_temp = alpha * initial_temp  # decrement temperature
            print("Temp: " + str(initial_temp))
            print("Best Solution Score: " + str(best_solution_score))

        return best_solution

    def flip(self, lista2, i, j):  # flip operator
        lista1 = copy.copy(lista2)
        lista1[i], lista1[j] = lista2[j], lista2[i]

        return lista1

    def get_neighbours(self, solution):  # get adjacent neighbours
        all_neighbours = []
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                if j - i == 1:
                    neighbour = self.flip(solution, i, j)
                    all_neighbours.append(neighbour)

        return all_neighbours

    ### GENETIC ALGORITHM ###

    def genetic(self):
        population = self.generate_population(20)

        # For 100 new Generations
        for _ in range(100):

            # Reproduction
            offspring1 = self.reproduction(population, elitism)

            # Mutation
            offspring2 = self.mutation(offspring1)

            # Evaluate fitness of offspring
            population = offspring2

            # Sorts the list from the most fitness solution to the least fitness
            population.sort(key=self.evaluate, reverse=False)

        return population[0]

    # Generates a population with population_size possible solutions
    def generate_population(self, population_size):
        population = []

        for _ in range(population_size):
            population.append(self.generate_random_solution())

        return population

    # Generates a new population of constant size, selecting two parents at each iteration and the order-based crossover between them
    def reproduction(self, population, elitism):
        offspring = []
        size = len(population)

        if elitism == True:
            # Sorts the list from the most fitness solution to the least fitness and keeps the most fitness solution
            population.sort(key=self.evaluate, reverse=False)
            offspring.append(population[0])

        while len(offspring) < size:

            # Get parents
            parents = self.parent_selection(population)

            parent1 = parents[0]
            parent2 = parents[1]

            # Applies the order-based crossover and keeps the generated children
            child1 = self.crossover(parent1, parent2)
            offspring.append(child1)

            if elitism == False:
                child2 = self.crossover(parent2, parent1)
                offspring.append(child2)

        return offspring

    # Selects two parents using the method Tournament Selection
    def parent_selection(self, population):

        # Get parents
        parent1 = self.tournament_selection(population)
        parent2 = self.tournament_selection(population)

        # To avoid duplicates
        while(parent1 == parent2):
            parent2 = self.tournament_selection(population)

        return [parent1, parent2]

    # Gets an candidate with the method Tournament Selection
    def tournament_selection(self, population):
        size = len(population)

        population_ids = set([x for x in range(size)])
        candidates = random.sample(population_ids, 2)

        # Get to candidates
        candidate1 = population[candidates[0]]
        candidate2 = population[candidates[1]]

        if self.evaluate(candidate1) > self.evaluate(candidate2):
            return candidate1
        else:
            return candidate2

    # Uses a genetic operator of order-based crossover to generate a new solution
    def crossover(self, parent1, parent2):

        child = []
        size = len(parent1)
        half_size = size // 2
        start = random.randint(0, half_size)

        # Selection of a subset from parent1
        for i in range(size):
            if i >= start and i < start + half_size:
                child.append(parent1[i])
            else:
                child.append(None)

        # Remaining elements of parent1 order by parent2
        remaining = list(self.filter_list(parent2, child))

        # Filling the child with the remaining  elements
        j = 0
        for i in range(size):
            if child[i] == None:
                child[i] = remaining[j]
                j += 1

        return child

    # Applies the genetic operator of mutation in the new population, with a probability of 0.05
    def mutation(self, population):
        offspring = []

        for chromosome in population:

            chromosome_size = len(chromosome)

            if random.random() <= 0.05:
                position1 = random.randint(0, chromosome_size - 1)
                position2 = random.randint(0, chromosome_size - 1)

                # To avoid duplicates
                while position1 == position2:
                    position2 = random.randint(0, chromosome_size - 1)

                mutated_chromosome = self.flip(
                    chromosome, position1, position2)
                offspring.append(mutated_chromosome)
            else:
                offspring.append(chromosome)

        return offspring

    ### UTILS ###

    # Gets the remaining elements from the full_list
    def filter_list(self, full_list, excludes):
        s = set(excludes)

        return (x for x in full_list if x not in s)

    ### INITIAL SOLUTIONS ###

    # Chooses the type of initial approach
    def get_initial_solution(self):

        if solution_type == 1:
            return self.generate_random_solution()
        elif solution_type == 2:
            return self.generate_greedy_score_solution()
        elif solution_type == 3:
            return self.generate_greedy_signup_solution()

    # Generates a greedy solution that sorts the list of libraries from the library with the highest book score to the library with the lowest book score
    def generate_greedy_score_solution(self):
        solution = [[libraries[i].id, libraries[i].total_score]
                    for i in range(n_libraries)]

        sorted_solution = sorted(solution, key=lambda x: x[1], reverse=True)

        return [sol[0] for sol in sorted_solution]

    # Generates a greedy solution that sorts the list of libraries from the library with the lowest time of signup to the library with the highest time of signup
    def generate_greedy_signup_solution(self):
        solution = [[libraries[i].id, libraries[i].signup]
                    for i in range(n_libraries)]

        sorted_solution = sorted(solution, key=lambda x: x[1], reverse=False)

        return [sol[0] for sol in sorted_solution]

    # Generates a random solution
    def generate_random_solution(self):
        return random.sample(range(0, n_libraries), n_libraries)


def main():
    global solution_type
    global evaluation
    global algorithm
    global name

    print("")
    print("|------------- Input File ---------------|")
    print("|        1. a_example.txt                |")
    print("|        2. b_read_on.txt                |")
    print("|        3. c_incunabula.txt             |")
    print("|        4. d_tough_choices.txt          |")
    print("|        5. e_so_many_books.txt          |")
    print("|        6. f_livraries_of_the_world.txt |")
    print("")

    name_option = int(input("|   > Select  "))

    if name_option == 1:
        name += 'a_example.txt'
    elif name_option == 2:
        name += 'b_read_on.txt'
    elif name_option == 3:
        name += 'c_incunabula.txt'
    elif name_option == 4:
        name += 'd_tough_choices.txt'
    elif name_option == 5:
        name += 'e_so_many_books.txt'
    elif name_option == 6:
        name += 'f_livraries_of_the_world.txt'

    print("")
    print("|------------- Heuristic ----------------|")
    print("|        1. Real Score                   |")
    print("|        2. Score                        |")
    print("|        3. Heuristic                    |")
    print("")

    evaluation = int(input("|   > Select  "))

    print("")
    print("|------------- Algorithms ---------------|")
    print("|        1. First Accept Hill Climbing   |")
    print("|        2. Steepest Hill Climbing       |")
    print("|        3. Simulated Annealing          |")
    print("|        4. Genetic Algorithm            |")
    print("")

    algorithm = int(input("|   > Select  "))

    if algorithm ==1 or algorithm ==2 or algorithm ==3:
        print("")
        print("|------------- Initial Solution ---------|")
        print("|        1. Random                       |")
        print("|        2. Greedy Score                 |")
        print("|        3. Greedy Signup                |")
        print("")

        solution_type = int(input("|   > Select  "))

    solver = Solver(name)

    print("** Reading input **")
    solver.input()

    print("** Solving **")
    solution = solver.solve(algorithm)
    print("Best Score: " + str(solution))

    print("** Writing output **")
    solver.output()


if __name__ == '__main__':
    main()
