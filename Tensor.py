import numpy as np

class Tensor:

    def __init__(self, data=None, shape=None, children: set=(), operation="", label=None, requires_grad=True):
        assert data is not None or shape is not None
        assert data is None or shape is None
            
        if data is not None:
            if isinstance(data, list):
                self.data = np.array(data, dtype=float)
            elif isinstance(data, (int, float, np.float32, list)):
                self.data = np.array(data, dtype=float)
            else:
                assert isinstance(data, np.ndarray)
                self.data = data
        else:
            self.data = np.zeros(shape)
        self._requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(children)
        self._operation = operation
        self._label = label
        if self._requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __matmul__(self, other):
        other = Tensor(data=other) if not isinstance(other, Tensor) else other
        #out_data = self.data @ other.data
        out_data = np.dot(self.data, other.data)
        assert isinstance(other, Tensor)
        out = Tensor(data=out_data, children=(self, other), requires_grad=self._requires_grad)

        def _backward():
            #self.grad += out.grad @ np.atleast_2d(other.data)#.transpose()
            #other.grad += self.data.transpose() @ out.grad

            if len(self.data.shape) == 1 and len(other.data.shape) == 1:
                self.grad += np.dot(np.atleast_2d(out.grad).transpose(), np.atleast_2d(other.data))
                other.grad += np.dot(np.atleast_2d(self.data).transpose(), np.atleast_2d(out.grad))
            elif len(self.data.shape) == 1 and len(other.data.shape) == 2:
                self.grad += np.dot(np.atleast_2d(out.grad), other.data.transpose())
                other.grad += np.dot(np.atleast_2d(self.data).transpose(), out.grad)
            elif len(self.data.shape) == 2 and len(other.data.shape) == 1:
                self.grad += np.atleast_2d(out.grad).transpose() @ np.atleast_2d(other.data)
                other.grad += np.dot(self.data.transpose(), out.grad)
            else:
                assert len(self.data.shape) == 2 and len(other.data.shape) == 2
                self.grad += np.dot(out.grad, other.data.transpose())
                other.grad += np.dot(self.data.transpose(), out.grad)

        out._backward = _backward
        return out

    def scalar_mul(self, scalar):
        out_data = scalar * self.data
        out = Tensor(data=out_data, children=(self,), requires_grad=self._requires_grad)

        def _backward():
            self.grad += scalar * out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self.scalar_mul(other)
        if isinstance(other, np.ndarray):
            other = Tensor(data=other.astype(np.float)) 
        else:
            assert isinstance(other, Tensor)

        other = Tensor(data=other) if not isinstance(other, Tensor) else other
        out_data = self.data * other.data
        out = Tensor(data=out_data, children=(self, other), requires_grad=self._requires_grad)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(data=(other * np.ones(self.data.shape)))
        elif isinstance(other, np.ndarray):
            other = Tensor(data=other.astype(np.float)) 
        else:
            assert isinstance(other, Tensor)
        
        out_data = self.data + other.data
        out = Tensor(data=out_data, children=(self, other), requires_grad=self._requires_grad)

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        ones_data = Tensor(data = -np.ones(self.shape))
        return self * ones_data

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self - other
    
    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Only int/float can be used"
        out_data = self.data ** exponent
        out = Tensor(data=out_data, children=(self,), requires_grad=self._requires_grad)

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad.data
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
        out = Tensor(data=t, children=(self,), operation="tanh")
        
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out_data = np.maximum(0.0, self.data)
        out = Tensor(data=out_data, children=(self,), operation='ReLU')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def gelu(self):
        out_data = None
        out = Tensor(data=out_data, children=(self,), operation='GeLU')

        def _backward():
            pass

        out._backward = _backward
        return out

    def leaky_relu(self):
        out_data = None
        out = Tensor(data=out_data, children=(self,), operation='Leaky ReLU')

        def _backward():
            pass

        out._backward = _backward
        return out

    def softmax(self):
        max = np.max(self.data, keepdims=True, axis=-1)
        exp = np.exp(self.data - max)
        exp_sum = np.sum(exp, axis=-1, keepdims=True)
        s = exp / exp_sum

        #exp = np.exp(self.data)
        #exp_sum = np.sum(exp, dim=-1, keepdims=True)
        out = Tensor(data=s, children=(self,), operation="softmax")
        
        def _backward():
            J = np.diag(s) - np.outer(s, s)
            self.grad += J @ out.grad
        
        out._backward = _backward
        return out

    def cross_entropy(self, target):
        if isinstance(target, (int, np.int64)):
            target = np.eye(self.data.shape[-1])[target]
        assert isinstance(target, np.ndarray)
        cs = -np.sum(target * np.log(self.data), axis=-1, keepdims=False)
        out = Tensor(data = cs, children=(self,), operation="cross-entropy")

        def _backward():
            #self.grad += (target / cs + (1-target) / (1-cs)) * out.grad
            self.grad += -target / self.data * out.grad

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.einsum("ij->ji")

    # The main backward method
    def backward(self):
        assert self.data.shape == (1,) or self.data.shape == ()
        self.grad = np.ones_like(self.data)
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


from graphviz import Digraph

""" JUST ADDING THE .grad TO THE VISUALIZATION"""
def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_node_label(tensor):
  l = ""
  assert len(tensor.shape) == 0 or len(tensor.shape) == 1 or len(tensor.shape) == 2
  if len(tensor.shape) == 0:
    return f'{tensor.item():.2f}' 
  elif len(tensor.shape) == 1:
    return " ".join([f'{tensor.data[i]:.2f}' for i in range(tensor.shape[0])])
  else:
    for i in range(tensor.shape[0]):
      l += " ".join([f'{tensor.data[i, j]:.2f}' for j in range(tensor.shape[1])]) + "\\n"
    return l

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    label = '{' + f' {n._label} | ' + draw_node_label(n.data) + " | " + draw_node_label(n.grad) + '}'
    dot.node(name = uid, label = label, shape='record')
    if n._operation:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._operation, label = n._operation)
      # and connect this node to it
      dot.edge(uid + n._operation, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._operation)

  return dot