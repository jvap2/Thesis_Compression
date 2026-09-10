read_liberty /home/jvap2/eda/nangate45/lib/NangateOpenCellLibrary_typical.lib
read_verilog synth_out/mac3_shiftadd/shift_add_mac_net.v
link_design shift_add_mac
create_clock -name clk -period 1.0 [get_ports clk]
read_power_activities -scope tb_power/dut -vcd synth_out/mac3_shiftadd/power/power.vcd
report_power -digits 5
exit
