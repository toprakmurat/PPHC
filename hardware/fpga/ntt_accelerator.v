//-----------------------------------------------------------------------------
// SecurePulse-FHE: NTT Hardware Accelerator
//-----------------------------------------------------------------------------
// Module: ntt_accelerator
// Description: FPGA-based Number Theoretic Transform accelerator for FHE
//              Implements radix-2 Cooley-Tukey butterfly operations
//
// Target: Xilinx Alveo U250 / Intel Stratix 10
// Performance Goal: 10-100x speedup over CPU implementation
//-----------------------------------------------------------------------------

module ntt_accelerator #(
    parameter DATA_WIDTH     = 64,          // Coefficient bit width
    parameter LOG_N          = 12,          // log2(polynomial degree), N = 4096
    parameter MODULUS        = 64'hFFFFFFFF00000001,  // Prime modulus q
    parameter NUM_BUTTERFLIES = 16          // Parallel butterfly units
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control interface
    input  wire                     start,
    input  wire                     inverse_mode,   // 0: forward NTT, 1: inverse NTT
    output reg                      done,
    output reg                      busy,
    
    // Data input interface (AXI-Stream style)
    input  wire [DATA_WIDTH-1:0]    data_in,
    input  wire                     data_in_valid,
    output wire                     data_in_ready,
    
    // Data output interface
    output reg  [DATA_WIDTH-1:0]    data_out,
    output reg                      data_out_valid,
    input  wire                     data_out_ready,
    
    // Twiddle factor ROM interface
    output reg  [LOG_N-1:0]         twiddle_addr,
    input  wire [DATA_WIDTH-1:0]    twiddle_data
);

    //-------------------------------------------------------------------------
    // Local Parameters
    //-------------------------------------------------------------------------
    localparam N = 1 << LOG_N;
    localparam STAGES = LOG_N;
    
    localparam STATE_IDLE       = 3'b000;
    localparam STATE_LOAD       = 3'b001;
    localparam STATE_COMPUTE    = 3'b010;
    localparam STATE_STORE      = 3'b011;
    localparam STATE_DONE       = 3'b100;

    //-------------------------------------------------------------------------
    // Internal Signals
    //-------------------------------------------------------------------------
    reg [2:0]               state, next_state;
    reg [LOG_N:0]           load_counter;
    reg [LOG_N:0]           store_counter;
    reg [3:0]               stage_counter;
    reg [LOG_N-1:0]         butterfly_counter;
    
    // Coefficient memory (dual-port BRAM)
    reg [DATA_WIDTH-1:0]    coeff_mem [0:N-1];
    
    // Butterfly unit signals
    wire [DATA_WIDTH-1:0]   butterfly_a_in;
    wire [DATA_WIDTH-1:0]   butterfly_b_in;
    wire [DATA_WIDTH-1:0]   butterfly_a_out;
    wire [DATA_WIDTH-1:0]   butterfly_b_out;
    wire [DATA_WIDTH-1:0]   twiddle_factor;
    
    reg [LOG_N-1:0]         addr_a, addr_b;
    
    //-------------------------------------------------------------------------
    // FSM: State Register
    //-------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
        end else begin
            state <= next_state;
        end
    end

    //-------------------------------------------------------------------------
    // FSM: Next State Logic
    //-------------------------------------------------------------------------
    always @(*) begin
        next_state = state;
        case (state)
            STATE_IDLE: begin
                if (start) next_state = STATE_LOAD;
            end
            
            STATE_LOAD: begin
                if (load_counter == N) next_state = STATE_COMPUTE;
            end
            
            STATE_COMPUTE: begin
                if (stage_counter == STAGES && butterfly_counter == 0)
                    next_state = STATE_STORE;
            end
            
            STATE_STORE: begin
                if (store_counter == N) next_state = STATE_DONE;
            end
            
            STATE_DONE: begin
                next_state = STATE_IDLE;
            end
            
            default: next_state = STATE_IDLE;
        endcase
    end

    //-------------------------------------------------------------------------
    // Load Counter
    //-------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            load_counter <= 0;
        end else if (state == STATE_IDLE) begin
            load_counter <= 0;
        end else if (state == STATE_LOAD && data_in_valid && data_in_ready) begin
            load_counter <= load_counter + 1;
        end
    end

    //-------------------------------------------------------------------------
    // Coefficient Memory Write (Load Phase)
    //-------------------------------------------------------------------------
    always @(posedge clk) begin
        if (state == STATE_LOAD && data_in_valid && data_in_ready) begin
            coeff_mem[load_counter[LOG_N-1:0]] <= data_in;
        end
    end

    //-------------------------------------------------------------------------
    // Stage and Butterfly Counters
    //-------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage_counter <= 0;
            butterfly_counter <= 0;
        end else if (state == STATE_IDLE) begin
            stage_counter <= 0;
            butterfly_counter <= 0;
        end else if (state == STATE_COMPUTE) begin
            if (butterfly_counter == (N >> 1) - 1) begin
                butterfly_counter <= 0;
                stage_counter <= stage_counter + 1;
            end else begin
                butterfly_counter <= butterfly_counter + 1;
            end
        end
    end

    //-------------------------------------------------------------------------
    // Butterfly Address Generation
    //-------------------------------------------------------------------------
    always @(*) begin
        // Cooley-Tukey decimation-in-time addressing
        // TODO: Implement proper bit-reversal and address computation
        addr_a = butterfly_counter;
        addr_b = butterfly_counter + (N >> 1);
    end

    //-------------------------------------------------------------------------
    // Twiddle Factor Address
    //-------------------------------------------------------------------------
    always @(posedge clk) begin
        if (state == STATE_COMPUTE) begin
            twiddle_addr <= (butterfly_counter << stage_counter);
        end else begin
            twiddle_addr <= 0;
        end
    end

    //-------------------------------------------------------------------------
    // Butterfly Unit (Modular Arithmetic)
    //-------------------------------------------------------------------------
    assign butterfly_a_in = coeff_mem[addr_a];
    assign butterfly_b_in = coeff_mem[addr_b];
    assign twiddle_factor = inverse_mode ? twiddle_data : twiddle_data; // TODO: inverse twiddles

    butterfly_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .MODULUS(MODULUS)
    ) u_butterfly (
        .clk(clk),
        .a_in(butterfly_a_in),
        .b_in(butterfly_b_in),
        .twiddle(twiddle_factor),
        .a_out(butterfly_a_out),
        .b_out(butterfly_b_out)
    );

    //-------------------------------------------------------------------------
    // Store Counter and Output
    //-------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            store_counter <= 0;
        end else if (state == STATE_IDLE) begin
            store_counter <= 0;
        end else if (state == STATE_STORE && data_out_valid && data_out_ready) begin
            store_counter <= store_counter + 1;
        end
    end

    always @(posedge clk) begin
        if (state == STATE_STORE) begin
            data_out <= coeff_mem[store_counter[LOG_N-1:0]];
        end
    end

    //-------------------------------------------------------------------------
    // Output Signals
    //-------------------------------------------------------------------------
    assign data_in_ready = (state == STATE_LOAD);
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out_valid <= 1'b0;
            done <= 1'b0;
            busy <= 1'b0;
        end else begin
            data_out_valid <= (state == STATE_STORE);
            done <= (state == STATE_DONE);
            busy <= (state != STATE_IDLE);
        end
    end

endmodule


//-----------------------------------------------------------------------------
// Butterfly Unit: Modular Multiplication and Addition
//-----------------------------------------------------------------------------
module butterfly_unit #(
    parameter DATA_WIDTH = 64,
    parameter MODULUS    = 64'hFFFFFFFF00000001
)(
    input  wire                     clk,
    input  wire [DATA_WIDTH-1:0]    a_in,
    input  wire [DATA_WIDTH-1:0]    b_in,
    input  wire [DATA_WIDTH-1:0]    twiddle,
    output reg  [DATA_WIDTH-1:0]    a_out,
    output reg  [DATA_WIDTH-1:0]    b_out
);
    
    // Pipeline registers
    reg [2*DATA_WIDTH-1:0] mult_result;
    reg [DATA_WIDTH-1:0]   b_twiddle;
    
    // Modular multiplication: b * twiddle mod q
    always @(posedge clk) begin
        mult_result <= b_in * twiddle;
    end
    
    // Barrett reduction (simplified placeholder)
    always @(posedge clk) begin
        b_twiddle <= mult_result % MODULUS;  // Synthesizable placeholder
    end
    
    // Butterfly outputs: a' = a + b*w, b' = a - b*w (mod q)
    always @(posedge clk) begin
        if ((a_in + b_twiddle) >= MODULUS)
            a_out <= a_in + b_twiddle - MODULUS;
        else
            a_out <= a_in + b_twiddle;
            
        if (a_in >= b_twiddle)
            b_out <= a_in - b_twiddle;
        else
            b_out <= a_in - b_twiddle + MODULUS;
    end

endmodule
