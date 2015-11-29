import numpy as np

def main():
    
    # given alpha value
    alpha = 0.85
    
    # read text from given data
    lines = read_data("data.txt")    
    
    # get node and edge 
    first_line = lines[0]
    node = int(first_line[0:first_line.index('\t')])
    edge = int(first_line[first_line.index('\t')+1:first_line.index('\n')])    
    
    # get edges in matrix form
    edge_mat = get_edge_mat(lines[1:])            
    
    # get adjacency matrix 
    adj_mat = get_adj_mat(node, edge_mat)   
    
    # get transition probability matrix
    tran_prob_mat = get_prob_mat(adj_mat)    
    
    # integrate teleporting (smoothing)
    mat_with_teleport = get_mat_with_teleport(tran_prob_mat,alpha,node)
        
    # get pagerank values
    init_distribution = (np.ones(node)/node).tolist()
    pagerank_vals = get_pagerank(mat_with_teleport, init_distribution)
    print(pagerank_vals)  

def read_data(file_name):
    f = open(file_name, 'r') # open file_name
    lines = f.readlines()    # store data into lines               
    f.close()                # close file_name
    return lines             # return data
    
def get_edge_mat(data):
    tmp_matrix = []          # initialize tmp_matrix
    num_of_spaces = data[0].strip().count(' ')    # count # of spaces after stripping blanks on sides
    for datum in data:       
        n = int(datum.strip()[0:datum.strip().find(' ')]) # store node in n
        e = int(datum.strip()[num_of_spaces:])            # store edge in e
        tmp = [n,e]                               # store [node,edge] in tmp
        tmp_matrix.append(tmp)                    # add [node,edge] to tmp_matrix
        
    return tmp_matrix
    
def get_adj_mat(num_of_node, edge_matrix):    # get adjacency matrix
    adj_matrix = np.zeros((num_of_node,num_of_node)) # get matrix of zeros
    for i in edge_matrix: 
        adj_matrix[i[0]-1,i[1]-1] = adj_matrix[i[0]-1,i[1]-1] + 1 # add 1 to places that have edge(s)
    return adj_matrix
    
def get_prob_mat(adj_matrix):
    prob_matrix = []                        
    for i in adj_matrix:                   
        if (sum(i)!=0):     # calculate # of edges and if not zero
            prob_matrix.append(i/sum(i))  # divide all elements by total # and add to prob_matrix
        else:               # if there are no edges at all
            prob_matrix.append(i/1)       # then divide all elements(0's) by 1 and add to prob_matrix
    prob_matrix = np.array(prob_matrix)   # make prob_matrix into numpy array
    return prob_matrix

def get_mat_with_teleport(prob_mat, alpha, node):  
    prob_mat = prob_mat + ((1-alpha)/node) # add (.15/node) to each element for teleporting
    prob_mat = prob_mat / (1+(1-alpha))    # divide each element by (1.15)
    return prob_mat
    
def get_pagerank(tr_matrix, initial_distribution):
    vector_array = []             # initialize vector_array as empty vector
    vector_array.append(initial_distribution)   # add initial_distribution vector into vector_array
    vector_array.append((np.dot(initial_distribution,tr_matrix).tolist())) # add dot product result
    cnt = 0                       # initialize cnt to count # of iterations needed to getpagerank
    
    # execute matrix multiplication until x is the SAME (x=xP)
    while(vector_array[-2]!=vector_array[-1]): 
        vector_array.append((np.dot(vector_array[-1],tr_matrix)).tolist()) 
        cnt = cnt + 1    
    print('Number of iterations: %d'% cnt)
    print('Pagerank for each page:')
    return vector_array[-1]       

if __name__ == "__main__":
    main()    
