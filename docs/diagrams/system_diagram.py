import graphviz


dot = graphviz.Digraph(comment='GIADog system overview')

dot.render('output/system.gv', view=True)