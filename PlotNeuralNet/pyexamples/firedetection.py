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
    to_ConvConvRelu(
        name="ghost1",
        s_filer=256,
        n_filer=(16, 16),
        offset="(0,0,0)",
        to="(0,0,0)",
        width=(1.5, 1.5),
        height=32,
        depth=32,
        caption="Ghost-1",
    ),
    to_Pool(
        name="pool1", offset="(0,0,0)", to="(ghost1-east)", width=1, height=24, depth=24
    ),
    to_ConvConvRelu(
        name="ghost2",
        s_filer=128,
        n_filer=(32, 32),
        offset="(1.5,0,0)",
        to="(pool1-east)",
        width=(2, 2),
        height=24,
        depth=24,
        caption="Ghost-2",
    ),
    to_Pool(
        name="pool2", offset="(0,0,0)", to="(ghost2-east)", width=1, height=16, depth=16
    ),
    to_Conv(
        name="attention1",
        s_filer=128,
        offset="(1.8,0,0)",
        to="(pool2-east)",
        width=1.2,
        height=16,
        depth=16,
        caption="Attention",
    ),
    to_connection("pool2", "attention1"),
    # --- Temporal ConvLSTM Module ---
    to_ConvConvRelu(
        name="conv_lstm",
        s_filer=64,
        n_filer=(64, 64),
        offset="(2.0,0,0)",
        to="(attention1-east)",
        width=(2.5, 2.5),
        height=16,
        depth=16,
        caption="ConvLSTM\\n(3 Frames)",
    ),
    to_connection("attention1", "conv_lstm"),
    # --- Decoder / Upsample ---
    *block_Unconv(
        name="up1",
        botton="conv_lstm",
        top="up1_out",
        s_filer=128,
        n_filer=32,
        offset="(2.2,0,0)",
        size=(24, 24, 3.0),
        opacity=0.6,
    ),
    to_skip(of="ghost2", to="ccr_res_up1", pos=1.25),
    *block_Unconv(
        name="up2",
        botton="up1_out",
        top="up2_out",
        s_filer=256,
        n_filer=16,
        offset="(2.2,0,0)",
        size=(32, 32, 2.0),
        opacity=0.6,
    ),
    to_skip(of="ghost1", to="ccr_res_up2", pos=1.25),
    # --- Detection Head ---
    to_ConvConvRelu(
        name="head",
        s_filer=256,
        n_filer=(32, 2),
        offset="(1.5,0,0)",
        to="(up2_out-east)",
        width=(1.5, 1),
        height=32,
        depth=32,
        caption="Fire Head",
    ),
    to_connection("up2_out", "head"),
    # Output
    # to_input("output_fire.png", name="output"),
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")


if __name__ == "__main__":
    main()
