// SecurePulse-FHE: Twiddle Factor ROM for NTT Accelerator
// Stores precomputed powers of root of unity

module twiddle_rom #(
    parameter DATA_WIDTH = 64,
    parameter ADDR_WIDTH = 12
)(
    input  wire                    clk,
    input  wire [ADDR_WIDTH-1:0]   addr,
    output reg  [DATA_WIDTH-1:0]   data
);
    reg [DATA_WIDTH-1:0] rom [0:(1<<ADDR_WIDTH)-1];
    
    initial begin
        $readmemh("twiddle_factors.hex", rom);
    end
    
    always @(posedge clk) begin
        data <= rom[addr];
    end
endmodule
