read_liberty /home/jvap2/eda/nangate45/lib/NangateOpenCellLibrary_typical.lib
read_verilog synth_out/mac3_shiftadd/power/cores/emb1_core_net.v
link_design emb1_core
create_clock -name vclk -period 1.0
set_input_delay -clock vclk 0 [all_inputs]
read_power_activities -scope tb/d -vcd synth_out/mac3_shiftadd/power/cores/emb1_core.vcd
report_power -digits 5
exit
