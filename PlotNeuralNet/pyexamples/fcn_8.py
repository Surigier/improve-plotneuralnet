import sys

sys.path.append("../")
from pycore.tikzeng import *
from pycore.blocks import *


arch = [
    to_head(".."),
    to_cor(),
    to_begin(),
    # Input
    # to_input("thermal_sequence.png"),
    # --- GhostNet Backbone ---
    # input image
    # input image
    # to_input("cats.jpg"),
    # conv1
    to_ConvConvRelu(
        name="cr1",
        s_filer=256,
        n_filer=(64, 64),
        offset="(0,0,0)",
        to="(0,0,0)",
        width=(2, 2),
        height=40,
        depth=40,
        caption="conv1",
    ),
    to_Pool(
        name="p1",
        offset="(0,0,0)",
        to="(cr1-east)",
        width=1,
        height=35,
        depth=35,
        opacity=0.5,
    ),
    # conv2
    to_ConvConvRelu(
        name="cr2",
        s_filer=128,
        n_filer=(128, 128),
        offset="(1,0,0)",
        to="(p1-east)",
        width=(3, 3),
        height=35,
        depth=35,
        caption="conv2",
    ),
    to_Pool(
        name="p2",
        offset="(0,0,0)",
        to="(cr2-east)",
        width=1,
        height=30,
        depth=30,
        opacity=0.5,
    ),
    # conv3
    to_ConvConvRelu(
        name="cr3",
        s_filer=64,
        n_filer=(256, 256, 256),
        offset="(1,0,0)",
        to="(p2-east)",
        width=(4, 4),
        height=30,
        depth=30,
        caption="conv3",
    ),
    to_Pool(
        name="p3",
        offset="(0,0,0)",
        to="(cr3-east)",
        width=1,
        height=23,
        depth=23,
        opacity=0.5,
    ),
    # conv4
    to_ConvConvRelu(
        name="cr4",
        s_filer=32,
        n_filer=(512, 512, 512),
        offset="(1,0,0)",
        to="(p3-east)",
        width=(6, 6),
        height=23,
        depth=23,
        caption="conv4",
    ),
    to_Pool(
        name="p4",
        offset="(0,0,0)",
        to="(cr4-east)",
        width=1,
        height=15,
        depth=15,
        opacity=0.5,
    ),
    # conv5
    to_ConvConvRelu(
        name="cr5",
        s_filer=16,
        n_filer=(512, 512, 512),
        offset="(1,0,0)",
        to="(p4-east)",
        width=(6, 6),
        height=15,
        depth=15,
        caption="conv5",
    ),
    to_Pool(
        name="p5",
        offset="(0,0,0)",
        to="(cr5-east)",
        width=1,
        height=10,
        depth=10,
        opacity=0.5,
    ),
    # fc-conv
    to_ConvConvRelu(
        name="cr6_7",
        s_filer=8,
        n_filer=(4096, 4096),
        offset="(1,0,0)",
        to="(p5-east)",
        width=(10, 10),
        height=10,
        depth=10,
        caption="fc-conv",
    ),
    # score32
    to_Conv(
        name="score32",
        s_filer=8,
        n_filer=21,
        offset="(1,0,0)",
        to="(cr6_7-east)",
        width=2,
        height=10,
        depth=10,
        caption="score32",
        xlabel="K",
    ),
    # up32
    to_UnPool(
        s_filer=16,
        name="up32",
        offset="(1,0,0)",
        to="(score32-east)",
        width=2,
        height=15,
        depth=15,
        caption="up32",
        xlabel="K",
    ),
    # skip connection from conv4 (score16)
    to_Conv(
        name="score16",
        s_filer=16,
        n_filer=21,
        offset="(3,-4,0)",
        to="(cr4-south)",
        width=2,
        height=15,
        depth=15,
        caption="score16",
        xlabel="K",
    ),
    # add1
    to_Sum(name="add1", offset="(1.5,0,0)", to="(up32-east)", radius=2.5),
    # up16
    to_UnPool(
        s_filer=32,
        name="up16",
        offset="(1,0,0)",
        to="(add1-east)",
        width=2,
        height=23,
        depth=23,
        caption="up16",
        xlabel="K",
    ),
    # skip connection from conv3 (score8)
    to_Conv(
        name="score8",
        s_filer=32,
        n_filer=21,
        offset="(15,-5,0)",
        to="(cr3-south)",
        width=2,
        height=23,
        depth=23,
        caption="score8",
        xlabel="K",
    ),
    # add2
    to_Sum(name="add2", offset="(1.5,0,0)", to="(up16-east)", radius=2.5),
    # up8
    to_UnPool(
        s_filer=64,
        name="up8",
        offset="(2,0,0)",
        to="(add2-east)",
        width=2,
        height=40,
        depth=40,
        caption="up8",
        xlabel="K",
    ),
    # softmax
    to_SoftMax(
        name="softmax",
        s_filer=256,
        offset="(1,0,0)",
        to="(up8-east)",
        width=2,
        height=40,
        depth=40,
        caption="softmax",
    ),
    # to_input("output_mask.png"),
    # connections (key part)
    # to_connection("cr1", "p1"),
    to_connection("p1", "cr2"),
    to_connection("p2", "cr3"),
    to_connection("p3", "cr4"),
    to_connection("p4", "cr5"),
    to_connection("p5", "cr6_7"),
    to_connection("cr6_7", "score32"),
    to_connection("score32", "up32"),
    to_connection("up32", "add1"),
    to_connection_rightangle("score16", "add1",to_anchor="south",x_offset=10.8),
    to_connection("add1", "up16"),
    to_connection("up16", "add2"),
    to_connection_rightangle("score8", "add2", to_anchor="south", x_offset=5.5),
    to_connection("add2", "up8"),
    to_connection("up8", "softmax"),
    # skip connections
    to_connection_rightangle("cr4", "score16",y_offset=-6.3),
    to_connection_rightangle("cr3", "score8",y_offset=-8),
    # end
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")


if __name__ == "__main__":
    main()
