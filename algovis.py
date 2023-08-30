import inspect
import functools 
import ast
from matplotlib import pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import networkx as nx
import numpy as np

def format_code(src,lineno):
    ### src is a list of lines
    res = ""
    for i in range(len(src)):
        res += '{:2d} '.format(i+1)
        if i == lineno:
            res += "-> "
        else:
            res += "   "
        res += str(src[i])
    return res

def snapshot_frame(x):
    filename = x.f_code.co_filename
    startline = x.f_code.co_firstlineno
    thisline = x.f_lineno
    lineno = thisline-startline
    src = inspect.getsourcelines(x)[0]
    lcls = x.f_locals
    co = x.f_code
    name = co.co_name
    #return (filename,str(src[lineno]))
    return (filename,name + str(lcls)+") " + str(src[lineno]).strip())

def snapshot_code(level=1):
    ### returns (code,lineindex,arrayvars,matrixvars,strvars,listvars)
    ### here arrayvars ia a map from array variable names to pairs (arrayval,lst) where
    ### arrayval is the value of the array and lst is a list of pairs (var,val)
    ### where each var is an idex variable for that array and val is its value
    ### here listvars ia a map from list variable names to pairs (listval,lst) where
    ### listval is the value of the list and lst is a list of pairs (var,val)
    ### where each var is a pointer variable for that list and val is its value
    f = inspect.currentframe()
    while (level > 0):
        f = f.f_back
        level -= 1
    x = f
    startline = x.f_code.co_firstlineno
    thisline = x.f_lineno
    lineno = thisline-startline
    src = inspect.getsourcelines(x)[0]
    srctext = functools.reduce(lambda x, y: x+y,src)
    
    stacksnapshot = []
    curframe = f
    (curfile,cursnapshot) = snapshot_frame(curframe)
    while curframe != None and curfile.find("ipy") != -1:
        stacksnapshot.append(cursnapshot.strip())
        curframe = curframe.f_back
        (curfile,cursnapshot) = snapshot_frame(curframe)

    
    a = ast.parse(srctext)
    # print(ast.dump(a))
    lcls = x.f_locals.items()
    arrayvars = {}
    listvars = {}
    matrixvars = {}
    strvars = {}
    graphvars = {}
    treevars = {}

    for (var,val) in lcls:
        if srctext.find(' '+var+' is an array') != -1:
            arrayvars[var] = (val,[])
        if srctext.find(' '+var+' is a linked list') != -1:
            listvars[var] = (val,[])
        if srctext.find(' '+var+' is a matrix') != -1:
            matrixvars[var] = (val,[])
        if srctext.find(' '+var+' is a character array') != -1:
            strvars[var] = (val,[])
        if srctext.find(' '+var+' is a graph (adjacency matrix)') != -1:
            graphvars[var] = AdjM2Graph(val)
        if srctext.find(' '+var+' is a graph (adjacency list)') != -1:
            graphvars[var] = AdjL2Graph(val)
        elif srctext.find(' '+var+' is a tree') != -1:
            treevars[var] = (val,[])
    for var in arrayvars.keys():
        for (ivar,ival) in lcls:
            if srctext.find('%s indexes %s' % (ivar,var)) != -1:
                (val,lst) = arrayvars[var]
                lst.append((ivar,ival))
    for var in strvars.keys():
        for (ivar,ival) in lcls:
            if srctext.find('%s indexes %s' % (ivar,var)) != -1:
                (val,lst) = strvars[var]
                lst.append((ivar,ival))
    for var in listvars.keys():
        for (ivar,ival) in lcls:
            if srctext.find('%s points into %s' % (ivar,var)) != -1:
                (val,lst) = listvars[var]
                lst.append((ivar,ival))
    for var in matrixvars.keys():
        for (ivar,ival) in lcls:
            if srctext.find('%s indexes rows of %s' % (ivar,var)) != -1:
                (val,lst) = matrixvars[var]
                rivar = 'R:' + ivar
                lst.append((rivar,ival))
            if srctext.find('%s indexes cols of %s' % (ivar,var)) != -1:
                (val,lst) = matrixvars[var]
                civar = 'C:' + ivar
                lst.append((civar,ival))
    # label each node in the graph with variables that index it
    for var in graphvars.keys():
        mapping = {}
        for (ivar,ival) in lcls:
            if srctext.find('%s indexes nodes of %s' % (ivar,var)) != -1:
                if ival in mapping:
                    mapping[ival] = mapping[ival]+' %s' % ivar[0]
                else:
                    mapping[ival] = str(ival)+ ' %s' % ivar[0]
        graphvars[var] = nx.relabel_nodes(graphvars[var],mapping)
    for var in treevars.keys():
        for (ivar,ival) in lcls:
            if srctext.find('%s is currently being traversed' % (ivar)) != -1:
                (val,lst) = treevars[var]
                lst.append((ivar,ival))
    return (src,lineno,arrayvars,listvars,matrixvars,strvars,graphvars,treevars,lcls,stacksnapshot)

def format_list(L,ptrs,indent=0):
    ### print a snapshot of a linked list with various pointers
    indspos = []
    p = L
    line = ""
    while p != None:
        line += "-->|"
        extra = ' {:2d} '.format(p.val)
        for var,val in ptrs:
            if val == p:
                indspos.append((var,indent+len(line)+len(extra)/2))
        line += extra
        line += "||"
        p = p.next
    line2s = []
    line2index = 0
    while len(indspos) > 0:
        line2 = ' ' * indent
        upto = -1
        for i in range(indent,len(line)+indent):
            add = " "
            for (var,pos) in indspos:
                if i == pos and upto < i:
                    add = var[0]
                    indspos.remove((var,pos))
                    upto = i               
            line2 += add
        line2 += "\n"
        line2s.append(line2)
        line2index += 1
    output = ""
    # add pointer arrows to the index pointers
    for line2 in line2s:
        line2a = ""
        for ch in line2:
            if ch != ' ' and ch != '\n':
                line2a += '^'
            else:
                line2a += ch
        output += line2a
        output += line2
    
    return line + "\n" + output + "\n"

def format_array(A,inds,formatstr=' {:2d} '):
    ### print a snapshot of an array with various indices
    indspos = []
    line = "|"
    for i in range(0,len(A)):
        extra = formatstr.format(A[i])
        for var,val in inds:
            if val == i:
                indspos.append((var,len(line)+len(extra)/2))
        line += extra
        line += "|"
    line2 = ""
    while len(indspos) > 0:
        upto = -1
        for i in range(0,len(line)):
            add = " "
            for (var,pos) in indspos:
                if i == int(pos) and upto < i:
                    add = var[0]
                    indspos.remove((var,pos))
                    upto = i               
            line2 += add
        line2 += "\n"
    return line + "\n" + line2 + "\n"

def format_tree(root):
    def display(root):
        if root is None:
            line = ""
            width = 0
            height = 1
            middle = 0
            return [line], width, height, middle
            
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if root.right is None and root.left is None:
            line = '%s' % root.val
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if root.right is None:
            lines, n, p, x = display(root.left)
            s = '%s' % root.val
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if root.left is None:
            lines, n, p, x = display(root.right)
            s = '%s' % root.val
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = display(root.left)
        right, m, q, y = display(root.right)
        s = '%s' % root.val
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    lines, *_ = display(root)
    str = ""
    for line in lines:
        str += line + "\n"
    return str + "\n"


def format_matrix(M,inds,indent=0):
    ### print a snapshot of an matrix with indices
    #print('M:',M)
    #print('inds:',inds)
    lineindent = ' ' * indent
    indsposm = []
    line = "|"
    flagR = []
    flagC = []
    for i in range(0,len(M)):
        flagR = []
        for j in range(0,len(M[0])):
            extra = ' {:4d} '.format(M[i][j])            
            for var,val in inds:
                if var.find('R:') != -1 and var not in flagR:
                    if val == i:
                        flagR.append(var)
                elif var.find('C:') != -1 and var not in flagC:
                    if val == j:
                        indsposm.append((var,indent+len(line)+len(extra)/2))
                        flagC.append(var)
            line += extra
        line += "|"
        if i == 0:
            firstline = line
        for var in flagR:
            line += (var[2:] + " ")
        if i != len(M)-1:
            line += "\n" + lineindent + "|"
    
    line2 = ""
    while len(indsposm) > 0:
        upto = -1
        for i in range(0,(len(firstline))*2):
            add = " "
            for (var,pos) in indsposm:
                if i == int(pos) and upto < i:
                    add = var[2]
                    indsposm.remove((var,pos))
                    upto = i               
            line2 += add
        line2 += "\n"    
    
    #print(line)
    return line + "\n" + line2 + "\n"

frames = []
def clear_frames():
    global frames
    frames = []

def snapshot(printstack=False):
    global frames
    #print(*args)
    res = ""
    (src,lineno,arrayvars,listvars,matrixvars,strvars,graphvars,treevars,lcls,stacksnapshot) = snapshot_code(2)
    if printstack:
        res += "Stack: \n"
        for s in stacksnapshot:
            res += "   " + s + "\n"
        res += "\n"
    
    graph = None
    if len(graphvars) > 0:
        # XXX: only one graph at the moment
        key, value = graphvars.popitem()
        graph = value
    
    #print(matrixvars)
    for (var,val) in lcls:
        # print('var,val:', var,val)
        res += var +': '+ str(val) + '  '
    res += "\n\n"
    for (aname,(aval,indsvars)) in arrayvars.items():
        res += "%s: \n" % aname
        res += format_array(aval,indsvars)
    for (aname,(aval,indsvars)) in strvars.items():
        res += "%s: \n" % aname
        res += format_array(aval,indsvars,' {} ')
    for (lname,(lval,indsvars)) in listvars.items():
        start = "%s: " % lname
        res += (start + format_list(lval,indsvars,len(start)))
    for (mname,(mval,indsvars)) in matrixvars.items():
        mstart = "%s: " % mname
        res += (mstart + format_matrix(mval,indsvars,len(mstart)))
    for (tname,(tval,indsvars)) in treevars.items():
        
        # Add \n after :
        tstart = "Current Node: %s\n%s:\n" % (indsvars, tname)
        res += (tstart + format_tree(tval))
        # print(res)
    res += format_code(src,lineno)
    #print(res)
    frames.append((res,graph))

def AdjM2Graph(adj_matrix):
    # Convert the normal 2D array into the numpy array
    adj_matrix = np.asarray(adj_matrix)

    # Into edge list
    # rows: [0 0 1 1 2 3 3 3]
    # cols: [1 3 0 3 3 0 1 2]
    rows, cols = np.where(adj_matrix == 1)

    # {[0, 1], [0, 3], [1, 0] ... [3, 2]}
    edge_list = zip(rows.tolist(), cols.tolist())
    
    # Init NetworkX Graph, Change to Graph() for undirected Graph
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    #print(list(nx.dfs_labeled_edges(G, source=0)))
    return G

def AdjL2Graph(adj_list):
    d = {}
    for i in range(len(adj_list)):
        d[i] = adj_list[i]
    return nx.from_dict_of_lists(d, create_using=nx.DiGraph)




class TextAnimation:
    # XXX: take an array of axs, one for each graph
    def __init__(self, fig, frames, ax=None):
        self._fig = fig
        self._frames = frames
        self._ax = ax
        self._fp = mpl.font_manager.FontProperties(family='monospace')

        
    def animate(self, step):
        # render the graph
        graph = self._frames[step][1]
        if self._ax != None and graph != None:
            self.HighlightNode(graph,[])

        # render text
        msg = self._frames[step][0]
        t = mpl.text.Text(0, 0, msg, transform=self._fig.transFigure,
                            figure=self._fig, verticalalignment='bottom',
                            horizontalalignment='left', fontsize=14,
                                fontproperties=self._fp)
        self._fig.texts.clear()
        self._fig.texts.append(t)
        
        
    def HighlightNode(self, G, nodes):
        self._ax.clear()
        # Layout, control the position of each node
        pos = nx.spring_layout(G, seed=0)
    
        M = G.number_of_edges()
        edge_colors = ["black"] * 4
        
        # Color map
        #cmap = plt.cm.plasma
    
        # Draw the nodes
        nx.draw_networkx_nodes(G, pos, ax=self._ax, nodelist=list(set(G) - set(nodes)), node_size=2000, node_color="white", edgecolors="black")
        nx.draw_networkx_nodes(G, pos, ax=self._ax, nodelist=nodes, node_size=2000, node_color="green", edgecolors="black")
    
        # Draw the edges 
        nx.draw_networkx_edges(G, pos, ax=self._ax, node_size=2000, arrowstyle="->", arrowsize=20, edge_color=edge_colors, width=2)

        # Draw the (node) labels
        nx.draw_networkx_labels(G, pos, ax=self._ax, font_color="black")


def show_animation(size):
    fig = plt.figure(figsize=size)
    # XXX: have the graphs stored as globals, a bit like frames
    #      the gridspec then has to have room for all of the graphs plus the text
    gs = fig.add_gridspec(2,1)
    ax = None
    if frames[0][1] != None:
        # XXX: create one ax for each graph to render
        ax = fig.add_subplot(gs[0,0])
    textfig = fig.add_subfigure(gs[1,0])
    # XXX: needs to take an array of axs: one for each graph
    ta = TextAnimation(textfig, frames, ax)
    fa = FuncAnimation(fig, ta.animate, frames=len(frames), interval=200)
    plt.close()
    return HTML(fa.to_jshtml())

