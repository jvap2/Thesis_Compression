read_liberty /home/jvap2/eda/nangate45/lib/NangateOpenCellLibrary_typical.lib
read_verilog synth_out/mac3_shiftadd/power/emb2_net.v
link_design gf4_pe_lutxlut
create_clock -name clk -period 1.0 [get_ports clk]
read_power_activities -scope tb_pe_power/dut -vcd synth_out/mac3_shiftadd/power/emb2.vcd
report_power -digits 5
exit
