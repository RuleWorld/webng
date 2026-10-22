/**
 * ssa_wrapper.cpp
 *
 * Thin C API wrapper around Network3's Gillespie SSA for use as a
 * Python-callable shared library (libssa.so) in WEBNG.
 *
 * Key functions:
 *   ssa_init         — load .net file, build dependency graph (once per worker)
 *   ssa_set_state    — set species counts before propagation
 *   ssa_run          — run SSA from t_start to t_end, endpoint only
 *   ssa_run_multi    — run SSA from t_start to t_end, sample at num_ts timepoints
 *   ssa_get_state    — read species counts after propagation
 *   ssa_reseed       — change random seed
 */

 extern "C" {
    #include <math.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    }
    
    #include <vector>
    #include <map>
    #include <string>
    #include <cmath>
    
    #include "network3.hh"
    #include "muParser.h"

    #include "util/rand.hh"
    
    // Module-level state
    static std::string g_last_error;
    static int g_n_species = 0;
    static int g_initialized = 0;
    static mu::Parser g_stop_condition;
    
    static Elt_array*  g_rates       = nullptr;
    static Elt_array*  g_species     = nullptr;
    static Group*      g_spec_groups = nullptr;
    static Rxn_array*  g_reactions   = nullptr;
    
    // Defined in network3_reset.cpp (see there for why both are needed):
    //   libssa_reset_network3: returns Network3's globals to their load-time state
    //   libssa_disable_onthefly_generation: stops Network3 waiting on stdin for
    //     BNG2.pl when a species with no reactions is first populated
    void libssa_reset_network3();
    void libssa_disable_onthefly_generation();

    static void set_error(const char* msg) {
        g_last_error = std::string(msg);
        fprintf(stderr, "[libssa] ERROR: %s\n", msg);
    }
    
    extern "C" {
    
    int ssa_init(const char* net_file, int seed, int gillespie_update_interval) {
        if (!net_file) { set_error("ssa_init: net_file is NULL"); return -1; }

        // Start from a clean slate, even if this process already initialized (or
        // tried to). Network3 accumulates global state across inits instead of
        // replacing it, so a second init used to corrupt memory and crash; see
        // network3_reset.cpp. On the first call this is a no-op.
        //
        // The previous network's arrays are deliberately NOT freed: Network3's
        // add_Rxn caches a raw pointer to the last reaction list across calls,
        // so freed memory reused for a new list could be written through that
        // stale pointer. Leaving them allocated (a small, one-time leak per
        // re-init) makes that impossible.
        libssa_reset_network3();
        g_reactions = nullptr; g_species = nullptr; g_rates = nullptr;
        g_spec_groups = nullptr;
        g_n_species = 0; g_initialized = 0;
    
        FILE* netfile = fopen(net_file, "r");
        if (!netfile) { set_error("ssa_init: cannot open net_file"); return -1; }
    
        int net_line_number = 0;
        int n_read = 0;
        double t = 0.0;
    
        g_rates = read_Elt_array(netfile, &net_line_number,
                                  (char*)"parameters", &n_read, nullptr);
        fprintf(stdout, "Read %d parameters from %s\n", n_read, net_file);
        rewind(netfile); net_line_number = 0;
    
        g_species = read_Elt_array(netfile, &net_line_number,
                                    (char*)"species", &n_read, g_rates);
        if (!g_species) { set_error("ssa_init: failed to read species"); fclose(netfile); return -1; }
        fprintf(stdout, "Read %d species from %s\n", n_read, net_file);
        rewind(netfile); net_line_number = 0;
    
        g_spec_groups = read_Groups(nullptr, netfile, g_species,
                                     &net_line_number, (char*)"groups", &n_read);
        fprintf(stdout, "Read %d group(s) from %s\n", n_read, net_file);
        rewind(netfile); net_line_number = 0;
    
        map<string, double*> param_map    = init_param_map(g_rates, g_spec_groups);
        map<string, int>     param_index  = init_param_index_map(g_rates);
        map<string, int>     observ_index = init_observ_index_map(g_spec_groups);
        read_functions_array(net_file, g_rates, param_map, param_index, observ_index, &t);
        int n_func = (int)network.functions.size();
        if (n_func > 0) n_func--;
        fprintf(stdout, "Read %d function(s) from %s\n", n_func, net_file);
    
        g_reactions = read_Rxn_array(netfile, &net_line_number, &n_read,
                                      g_species, g_rates, network.is_func_map);
        if (!g_reactions) { set_error("ssa_init: failed to read reactions"); fclose(netfile); return -1; }
        fprintf(stdout, "Read %d reaction(s) from %s\n", n_read, net_file);
        remove_zero_rate_rxns(&g_reactions, g_rates);
        fprintf(stdout, "%d reaction(s) have nonzero rate\n",
                g_reactions ? g_reactions->n_rxn : 0);
        fclose(netfile);
    
        char* network_name = strdup(net_file);
        chop_suffix(network_name, ".net");
        init_network(g_reactions, g_rates, g_species, g_spec_groups, network_name);
        free(network_name);
    
        for (int i = 0; i < network.species->n_elt; i++)
            network.species->elt[i]->val = floor(network.species->elt[i]->val + 0.5);
    
        int status = init_gillespie_direct_network(gillespie_update_interval, seed);
        if (status != 0) { set_error("ssa_init: init_gillespie_direct_network failed"); return -1; }
        libssa_disable_onthefly_generation();
    
        try { g_stop_condition.SetExpr("0"); }
        catch (mu::Parser::exception_type&) { set_error("ssa_init: stop condition failed"); return -1; }
    
        g_n_species = n_species_network();
        g_initialized = 1;
        fprintf(stdout, "libssa: initialized with %d species, %d reactions\n",
                g_n_species, g_reactions ? g_reactions->n_rxn : 0);
        return 0;
    }
    
    int ssa_set_state(double* species_counts, int n_species) {
        if (!g_initialized) { set_error("ssa_set_state: not initialized"); return -1; }
        if (n_species != g_n_species) { set_error("ssa_set_state: n_species mismatch"); return -1; }
        set_conc_network(species_counts);
        resync_gillespie_state();
        return 0;
    }
    
    /**
     * ssa_run: Run SSA from t_start to t_end. Endpoint only.
     * Use this for num_ts=2 (fastest).
     */
    int ssa_run(double t_start, double t_end) {
        if (!g_initialized) { set_error("ssa_run: not initialized"); return -1; }
        double t = t_start;
        double delta_t = t_end - t_start;
        int error = gillespie_direct_network(&t, delta_t, nullptr, nullptr, 1e15, g_stop_condition);
        return (error == 0 || error == -1) ? 0 : error;
    }
    
    /**
     * ssa_run_multi: Run one continuous SSA from t_start to t_end,
     * sampling species counts at num_ts evenly-spaced timepoints.
     *
     * This mirrors BioNetGen's simulate({method=>"ssa", t_end=>10, n_steps=>11})
     * behavior — one continuous random number stream, one dependency graph,
     * periodic snapshots at dt = (t_end - t_start) / (num_ts - 1).
     *
     * output: flattened (num_ts x n_species) array, row-major
     *         output[ts * n_species + i] = species i at timepoint ts
     * n_species: must match ssa_get_n_species()
     * num_ts: number of timepoints including t_start and t_end
     *
     * Returns 0 on success, -1 on error.
     */
    int ssa_run_multi(double t_start, double t_end, int num_ts,
                      double* output, int n_species) {
        if (!g_initialized) { set_error("ssa_run_multi: not initialized"); return -1; }
        if (n_species != g_n_species) { set_error("ssa_run_multi: n_species mismatch"); return -1; }
        if (num_ts < 2) { set_error("ssa_run_multi: num_ts must be >= 2"); return -1; }
    
        double dt = (t_end - t_start) / (double)(num_ts - 1);
    
        // Snapshot at t=0 (initial state before any SSA steps)
        get_conc_network(output);  // output[0 .. n_species-1]
    
        // Run SSA in num_ts-1 sub-intervals, snapshotting after each
        for (int ts = 1; ts < num_ts; ts++) {
            double t_i = t_start + (ts - 1) * dt;
            double t_f = t_start + ts * dt;
            double t   = t_i;
    
            // Run SSA for one sub-interval — continues from current network state
            int error = gillespie_direct_network(&t, dt, nullptr, nullptr, 1e15, g_stop_condition);
            if (error != 0 && error != -1) {
                set_error("ssa_run_multi: gillespie_direct_network failed");
                return error;
            }
    
            // Snapshot state at this timepoint
            double* row = output + ts * n_species;
            get_conc_network(row);
        }
    
        return 0;
    }
    
    int ssa_get_state(double* species_counts, int n_species) {
        if (!g_initialized) { set_error("ssa_get_state: not initialized"); return -1; }
        if (n_species != g_n_species) { set_error("ssa_get_state: n_species mismatch"); return -1; }
        get_conc_network(species_counts);
        return 0;
    }
    
    int ssa_get_n_species() { return g_n_species; }
    
    const char* ssa_get_last_error() { return g_last_error.c_str(); }
    
    void ssa_reseed(int seed) {
        if (g_initialized) {
            Util::SEED_RANDOM((unsigned long)seed);
        }
    }
    
    } // extern "C"