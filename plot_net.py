import pygraphviz as pgv
import json
import argparse
import os.path as path

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, required=True)
parser.add_argument('-o', type=str, default=None)
parser.add_argument('-v', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()

    verbose = args.v
    infile = args.i
    outfile = args.o

    if not path.exists(infile) or (outfile and not path.exists(outfile)):
        raise FileNotFoundError
    else:
        infile_path, infile_name = path.split(infile)
        infile_name = path.splitext(infile_name)[0]
        outfile = path.join(infile_path, infile_name + ".png")

    colors = {3: "#0382cd", 5: "#f9b72e", 7: "#a42035"}  # kernel size to color
    shapes = {i: 2*i/5 + 3/5 for i in range(1, 7)}  # expand ratio to shape
    dim = [1, 10, 51]  # C x H x W

    with open(infile, "r") as f:
        net = json.load(f)

    G = pgv.AGraph(directed=True)
    G.node_attr.update(shape="box", style="rounded,filled", fontname="arial bold", fontcolor="white", color="black", defaultdist=0.1)
    G.edge_attr.update(fontname="arial")

    G.add_node("Li", label="Input")
    G.add_edge("Li", "L0", label="x".join([str(d) for d in dim]))

    first_conv = net["first_conv"]
    conv_name = "Conv" if first_conv["name"] == "ConvLayer" else "Unknown"
    G.add_node("L0", color="#02afbc", label="{} {}x{}".format(conv_name, first_conv["kernel_size"][0], first_conv["kernel_size"][1]))
    dim[0] = first_conv["out_channels"]
    dim[1] //= first_conv["stride"][0]
    dim[2] //= first_conv["stride"][1]
    G.add_edge("L0", "L1", label="x".join([str(d) for d in dim]))

    layer = 1
    for block in net["blocks"]:
        prev_node = "L{}".format(layer-1)
        node = "L{}".format(layer)
        next_node = "L{}".format(layer + 1)
        conv = block["mobile_inverted_conv"]
        shortcut = block["shortcut"]

        if conv["name"] == "ZeroLayer":
            continue
        else:
            label = "MB{} {}x{}".format(conv["expand_ratio"], conv["kernel_size"], conv["kernel_size"])

        G.add_node(node, label=label, color=colors[conv["kernel_size"]], width=shapes[conv["expand_ratio"]])
        dim[0] = conv["out_channels"]
        dim[1] //= conv["stride"]
        dim[2] //= conv["stride"]
        G.add_edge(node, next_node, label="x".join([str(d) for d in dim]))

        if shortcut and shortcut["name"] == "IdentityLayer":
            G.add_edge(node, node, label="id")

        layer += 1

    if verbose:
        # Feature mix layer
        fmix_layer = net["feature_mix_layer"]
        conv_name = "Conv" if fmix_layer["name"] == "ConvLayer" else "Unknown"
        G.add_node("L{}".format(layer), color="#8b8c8d", label="{} {}x{}".format(conv_name, fmix_layer["kernel_size"], fmix_layer["kernel_size"]))
        dim[0] = fmix_layer["out_channels"]
        dim[1] //= fmix_layer["stride"]
        dim[2] //= fmix_layer["stride"]
        G.add_edge("L{}".format(layer), "L{}".format(layer+1), label="x".join([str(d) for d in dim]))
        layer += 1

        # Pooling
        G.add_node("L{}".format(layer), color="#8b8c8d", label="Avg. Pooling \n& flatten")
        dim[1] = 1
        dim[2] = 1
        G.add_edge("L{}".format(layer), "L{}".format(layer+1), label=str(dim[0]))
        layer += 1

        # Classifier
        classifier = net["classifier"]
        conv_name = "FC" if classifier["name"] == "LinearLayer" else "Unknown"
        G.add_node("L{}".format(layer), color="#8b8c8d", label=conv_name)
        dim[0] = classifier["out_features"]
        G.add_edge("L{}".format(layer), "L{}".format(layer+1), label=str(dim[0]))
        layer += 1
    else:
        # Classifier
        classifier = net["classifier"]
        conv_name = "FC" if classifier["name"] == "LinearLayer" else "Unknown"
        G.add_node("L{}".format(layer), color="#8b8c8d", label="Pooling {}".format(conv_name))
        dim[0] = classifier["out_features"]
        dim[1] = 1
        dim[2] = 1
        G.add_edge("L{}".format(layer), "L{}".format(layer+1), label=str(dim[0]))
        layer += 1

    # Output
    G.add_node("L{}".format(layer), label="Output")

    G.layout()
    G.draw(outfile, prog="dot")
