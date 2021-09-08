CC = sw5cc.new
HCFLAGS = -std=c99 -O3 -msimd -host 
SCFLAGS = -std=c99 -O3 -msimd -slave
LDFLAGS = -lm
HEADER_FILES = function.h args.h util.h
LINK_SPC = -Wl,--whole-archive,-wrap,athread_init,-wrap,__expt_handler,-wrap,__real_athread_spawn /home/export/online3/swmore/release/lib/libspc.a -Wl,--no-whole-archive

BUILD = ./obj

main : $(BUILD)/main.o $(BUILD)/function.o $(BUILD)/master.o $(BUILD)/slave.o
	$(CC) $(HCFLAGS) -hybrid -o $@ $^ $(LDFLAGS) -lm_slave $(LINK_SPC) 

$(BUILD)/%.o : %.c $(HEADER_FILES)
	$(CC) $(HCFLAGS)  -g  -o $@ -c $< 

$(BUILD)/slave.o : slave.c args.h
	$(CC) $(SCFLAGS)  -g  -o $@ -c $<

run : 
	bsub -I -b -q q_sw_expr -share_size 6144 -host_stack 1024 -n 1 -cgsp 64 ./main

clean :
	rm -f $(BUILD)/*.o ./main
