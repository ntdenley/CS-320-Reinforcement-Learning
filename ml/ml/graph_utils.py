from graphviz import Digraph
from .op import OpType

def to_string(node):
    string = "ComputeNode("
    string += f"\n\toptype = {node.op.optype}" if node.has_op else 'no op assigned'
    string += f"\n\tshape  = {node.shape}"
    string += f"\n\tdata   = {node.data}"
    return string + "\n)"
    
def to_dot(node, dot=None, view_data=True, color=None, depth=0, visited=[]):
    if node in visited: return
    if depth > 100: 
        print("view_graph warning: depth exceeded")
        return
    if dot is None: dot = Digraph(comment='Lazy Computation Graph')
    label = ""
    label += f"\noptype = {node.op.optype}" if node.has_op else 'no op assigned'
    label += f"\nshape  = {node.shape}"
    label += f"\nis_evaled = {node.is_evaled}"
    label += f"\ngrad = {node.grad}"
    if view_data:
        label += f"\ndata   = {node.data}"
    if not color:
        if node.has_op:
            if isinstance(node.op.optype, OpType.Alloc):
                color='lightgreen'
            elif isinstance(node.op.optype, OpType.View):
                color='lightblue'
            elif isinstance(node.op.optype, OpType.Inplace):
                color='lightgrey'
        else:
            color=None
    dot.node(name=str(id(node)), label=label, style='filled', color=color)
    visited.append(node)
    if node.has_op:
        for parent in node.op.get_parents():
            to_dot(parent, dot, view_data, depth=depth+1, visited=visited)
            dot.edge(str(id(parent)), str(id(node)))
    if node.grad and not isinstance(node.grad, str):
        to_dot(node.grad, dot, view_data, color="red", depth=depth+1, visited=visited)
        dot.edge(str(id(node)), str(id(node.grad)), label="grad", color="red")
    return dot

def view_graph(node, name="computation_graph", view=False, view_data=True):
    dot = to_dot(node, view_data=view_data, color="magenta")
    dot.render(filename=name, view=view)