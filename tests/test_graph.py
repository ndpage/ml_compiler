from ml_compiler.ir.graph import Graph, Node, GraphCycleError


def test_add_and_topological_order():
    g = Graph()
    a = Node(name="a", op_type="Input")
    b = Node(name="b", op_type="Relu", inputs=["a"])
    c = Node(name="c", op_type="Relu", inputs=["b"])
    for n in [a, b, c]:
        g.add_node(n)
    order = [n.name for n in g.topological_sort()]
    assert order == ["a", "b", "c"]


def test_cycle_detection():
    g = Graph()
    a = Node(name="a", op_type="Input", inputs=["c"])  # cycle a<-c
    b = Node(name="b", op_type="Relu", inputs=["a"])  # b<-a
    c = Node(name="c", op_type="Relu", inputs=["b"])  # c<-b forms cycle
    for n in [a, b, c]:
        g.add_node(n)
    try:
        g.topological_sort()
    except GraphCycleError:
        return
    assert False, "Cycle should raise GraphCycleError"
