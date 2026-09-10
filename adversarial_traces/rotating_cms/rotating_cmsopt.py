# (user-provided) class containing functions necessary for optimization:
#   gen traffic (as json), init function called each time we run interp iteration

import json
import pickle
import time
import random
import numpy as np

# helpers (useful if we're using real pcaps/traffic traces)
def i2Hex (n):
    hstr = str(hex(int(n)))
    if "0x" in hstr:
        if len(hstr[2:]) == 1: return "0" + hstr[2:].upper()
        else: return hstr[2:].upper()
    else:
        return hstr.upper()

def hexadecimal (ip_addr):
    # 0xC00002EB ( Hexadecimal )
    return "0x" + "".join( map( i2Hex,  ip_addr.split(".") ) )

class Opt:
    def __init__(self, config):
        # ground truth for entire trace
        self.ground_truth = {}
        # ground truth for most recent set of packets
        self.interval_ground_truth = {}
        # keep track of which cms we're cleaning (start writing to 0, cleaning 1)
        self.clean_sketch = 1
        # store the number of cms rows in program config
        self.rows = config["rows"]
        # store the number of cms cols in program config
        self.cols = config["cols"]

    # this gets called before each iteration
    # we can adjust the attributes of a trace by changing arguments
    def gen_traffic(self, traceparams):
        # reset ground truth for each interval
        self.interval_ground_truth.clear()
        # NOTE: we only use numpkts for random distribution, but could probably remove it altogether
        #### BACKGROUND TRAFFIC
        # number of background_pkts +/- 25%
        # sending same distribution as attack traffic for now, just smaller amount
        if traceparams["send_background"]:
            num_backgroundpkts = random.randint(int(.75*traceparams["background_pkts"]), int(1.25*traceparams["background_pkts"])) 
        else:
            num_backgroundpkts = 0
        ####
        num_attackpkts = traceparams["numpkts"]
        numpkts = num_attackpkts + num_backgroundpkts
        numflows = traceparams["numflows"]
        distribution = traceparams["distribution"]
        rate = traceparams["rate"]
        events = []
        # numpy can give us ints that are too large, so map them to smaller numbers
        skew_keys = {}

        # generate packets
        for p in range(numpkts):
            # for each packet, generate its header fields according to input arguments
            # e.g., if distribution = random, then we randomly select from numflows to set IPs
            #       if rate = 1000, then we send pkts every 1000ns (set timestamp = p*rate)
            if distribution == "random":  # generate trace w/ uniform header distribution
                ip_int = random.randint(1,numflows)
            elif distribution == "skew":    # generate trace w/ skewed zipfian header distr
                # TODO: setting default to distr 1.3 skew, can add this as a trace param later
                #   distr must be > 1, higher means more skewed
                ip_int = np.random.zipf(1.3)
                if ip_int not in skew_keys:
                    skew_keys[ip_int] = max(skew_keys.values(), default=0) + 1
                ip_int = skew_keys[ip_int]
            else:
                exit("distribution not implemented")

            #timestamp = prev_timestamp+((p+1)*rate)
            args = [ip_int, ip_int, 0]
            # NOTE: lucid interpreter acts weird if we don't specify a timestamp
            # it apparently matters how we set it????? not sure the general rule, but this seems to work for now
            pkt = {"name":"ip_in", "args":args, "timestamp": 100+((p+1)*rate)}
            events.append(pkt)
            if str(ip_int)+str(ip_int) not in self.ground_truth:
                self.ground_truth[str(ip_int)+str(ip_int)] = 1
            else:
                self.ground_truth[str(ip_int)+str(ip_int)] += 1
            if str(ip_int)+str(ip_int) not in self.interval_ground_truth:
                self.interval_ground_truth[str(ip_int)+str(ip_int)] = 1
            else:
                self.interval_ground_truth[str(ip_int)+str(ip_int)] += 1

        # dummy pkt, so we still write results to file
        if numpkts <= 0:
            events.append({"name": "ip_in", "args": [0,0,1], "timestamp": 101+rate})

        # update last dummy byte, this helps us identify the last pkt
        # (we call different extern on last pkt to write counts to file)
        events[-1]["args"][-1] = 1

        '''
        ##### CONTROL EVENTS TO CLEAN SKETCH
        ##### NOTE: we don't need this anymore, because we're doing it all in the data plane
        # add control events that cleans entire sketch
        #   for a single sketch, this could look something like:
        #       {"type": "command", "name":"Array.setrange", "args":{"array":"myarr", "start":0, "end":8,"value":[0]}}
        # or we can have rotating sketches that we switch off writing/cleaning from
        # NOTE: we can also set a range of array values instead one at a time, but the code is cleaner if we generate a control event for each cleaned register

        row_counter = 0
        col_counter = 0
        while numpkts > 0:
            if self.clean_sketch==0:
                events.append({"type": "command", "name":"Array.set", "args":{"array": "cms0.["+str(row_counter)+"]", "index": col_counter, "value": [0]}})
            elif self.clean_sketch==1:
                events.append({"type": "command", "name":"Array.set", "args":{"array": "cms1.["+str(row_counter)+"]", "index": col_counter, "value": [0]}}) 
            numpkts -= 1
            col_counter += 1
            if col_counter > self.cols - 1: # cleaned this entire row, move on to the next one
                col_counter = 0
                row_counter += 1

            if row_counter > self.rows - 1:   # we cleaned the entire sketch, so we're done (even if numpkts > 0)
                break

        # rotate sketches
        self.clean_sketch = 1 - self.clean_sketch
        '''

        #events_bytes = [(json.dumps(p)+'\n').encode('utf-8') for p in events]
        events_bytes = (json.dumps(events)+'\n').encode('utf-8')
        #print(self.interval_ground_truth)


        return events_bytes, num_backgroundpkts


    # # called after every interp run
    # # measurement is list of measurements (one measurement for each output file)
    # # order in list is same ordered specified in opt json
    # # avg_or_max is a list containing either "avg" or "max"
    # #   if ["avg"], then cost will be average error over all flows in a time window
    # #   if ["max"], cost will be max cost across all flows in a time window
    # def calc_cost(self, measure, avg_or_max=[], top_k=5, episode=None, step=None, action=None):  # compute avg error for our cms (mean abs error)
    #     m = measure[0]  # first file is estimated counts, second file is bits set

    #     # Sort ground truth by packet count (descending)
    #     sorted_gt = sorted(self.interval_ground_truth.items(), 
    #                     key=lambda item: item[1], 
    #                     reverse=True)
    #     # Take top K flows
    #     selected_flows = sorted_gt[:top_k]

    #     s = []

    #     total_abs_error = 0

    #     # Calculating relative error for all flows. Currently unused.
    #     # for k, v in self.interval_ground_truth.items():
    #     #     if abs(m[k]-self.ground_truth[k])/self.ground_truth[k] > 1:
    #     #         print("ERR!! ERROR > 1")
    #     #         print("est: "+str(m[k]))
    #     #         print("gt: "+str(self.ground_truth[k]))
    #     #         print("key: "+str(k))
    #     #         quit()
    #     #     s.append(abs(m[k] - v) / v)

    #     # REMOVED THE ASSERTION - keys may not match
    #     # assert m.keys() == self.interval_ground_truth.keys()

    #     # Calculating relative error for each flow in the top K flows
    #     for flow_key, true_count in selected_flows:
    #         # Get estimated count from sketch (default to 0 if not found)
    #         estimated_count = m.get(flow_key, 0)
            
    #         # Calculate absolute error
    #         abs_error = abs(estimated_count - true_count)
    #         total_abs_error += abs_error
            
    #         # Calculate relative error
    #         if true_count > 0:
    #             rel_error = abs_error / true_count
    #             s.append(rel_error)
    #         else:
    #             s.append(0)
        
    #     # for rank, (flow_key, true_count) in enumerate(selected_flows, 1):
    #     #     estimated_count = m.get(flow_key, 0)
    #     #     abs_error = abs(estimated_count - true_count)
    #     #     rel_error = abs_error / true_count if true_count > 0 else 0
            
    #     #     print(f"{rank:<6} {flow_key:<35} {true_count:<10} {estimated_count:<10} "
    #     #         f"{abs_error:<10.2f} {rel_error:<10.4f}")
        
    #     # Calculate and print MAE
    #     # mae = total_abs_error / len(selected_flows) if selected_flows else 0
    #     # avg_rel_error = sum(s) / len(s) if s else 0
        
    #     # print("-"*80)
    #     # print(f"MAE (Top {top_k}): {mae:.4f} packets")
    #     # print(f"Avg Relative Error: {avg_rel_error:.4f} ({avg_rel_error*100:.2f}%)")
    #     # print("="*80 + "\n")
        
    #     # Calculate final error
    #     if len(s) == 0:
    #         reward = 0
        
    #     if "max" in avg_or_max:
    #         reward = max(s)
    #     # return avg by default
    #     else:
    #         reward = sum(s) / len(s)
        
    #     # Write results to a fixed file if episode, step, and action are provided
    #     if episode is not None and step is not None and action is not None:
    #         # Append results to a fixed file
    #         with open("results.txt", "a") as f:
    #             # Write header only if file is empty (optional)
    #             if f.tell() == 0:
    #                 f.write("Episode\tStep\tAction\tReward\n")
    #             f.write(f"{episode}\t{step}\t{action}\t{reward:.4f}\n")
    #     # -------------------------------------------------

    #     return reward
    
    def calc_cost(self, measure, avg_or_max=[], top_k=5, episode=None, step=None, action=None):  # compute avg error for our cms (mean abs error)
        m = measure[0]  # first file is estimated counts, second file is bits set

        # Sort ground truth by packet count (descending)
        sorted_gt = sorted(self.interval_ground_truth.items(), 
                        key=lambda item: item[1], 
                        reverse=True)
        # Take top K flows
        selected_flows = sorted_gt[:top_k]

        s = []
        total_abs_error = 0

        # REMOVED THE ASSERTION - keys may not match
        # assert m.keys() == self.interval_ground_truth.keys()

        for flow_key, true_count in selected_flows:
            # Get estimated count from sketch (default to 0 if not found)
            estimated_count = m.get(flow_key, 0)
            
            # Calculate absolute error
            abs_error = abs(estimated_count - true_count)
            total_abs_error += abs_error
            
            # Calculate relative error
            if true_count > 0:
                rel_error = abs_error / true_count
                s.append(rel_error)
            else:
                s.append(0)

        # PRINT TOP K VALUES
        print("\n" + "="*80)
        print(f"TOP {top_k} FLOWS WITH ERRORS")
        print("="*80)
        print(f"{'Rank':<6} {'Flow ID':<35} {'True':<10} {'Est':<10} {'AbsErr':<10} {'RelErr':<10}")
        print("-"*80)
        
        for rank, (flow_key, true_count) in enumerate(selected_flows, 1):
            estimated_count = m.get(flow_key, 0)
            abs_error = abs(estimated_count - true_count)
            rel_error = abs_error / true_count if true_count > 0 else 0
            
            print(f"{rank:<6} {flow_key:<35} {true_count:<10} {estimated_count:<10} "
                f"{abs_error:<10.2f} {rel_error:<10.4f}")
        
        # Calculate and print MAE
        mae = total_abs_error / len(selected_flows) if selected_flows else 0
        avg_rel_error = sum(s) / len(s) if s else 0
        
        print("-"*80)
        print(f"MAE (Top {top_k}): {mae:.4f} packets")
        print(f"Avg Relative Error: {avg_rel_error:.4f} ({avg_rel_error*100:.2f}%)")
        print("="*80 + "\n")

        # --------------- NEW LOGGING PART ---------------
        if episode is not None and step is not None and action is not None:
            # Append results to a fixed file
            with open("results.txt", "a") as f:
                # Write header only if file is empty (optional)
                if f.tell() == 0:
                    f.write("Episode\tStep\tAction\tMAE\tAvgRelError\n")
                f.write(f"{episode}\t{step}\t{action}\t{mae:.4f}\t{avg_rel_error:.4f}\n")
        # -------------------------------------------------
        
        # Calculate final error
        if len(s) == 0:
            return 0
        
        if "max" in avg_or_max:
            return max(s)
        # return avg by default
        else:
            return sum(s) / len(s)


    # called before every interp run
    # create a blank json file with no events
    # (pass in events later in interactive mode)
    def init_simulation(self, maxpkts):
        info = {}
        info["switches"] = 1
        info["max time"] = maxpkts*1000
        info["default input gap"] = 1000
        info["random seed"] = 0
        info["python file"] = "rotating_cms.py"
        # REQUIRED FOR APPLE SILICON:
        # info["python path"] = "/opt/homebrew/Cellar/python@3.13/3.13.5/Frameworks/Python.framework/Versions/3.13/lib/libpython3.13.dylib"
        info["events"] = []
        with open('rotating_cms.json', 'w') as f:
            json.dump(info, f, indent=4)
        pass

'''
o = Opt("pcap")
o.gen_traffic()
#exit()
starttime = time.time()
cmd = ["make", "interp"]
ret = subprocess.run(cmd)

print("TIME", time.time()-starttime)
measurement = []
outfiles = ["test.pkl"]
for out in outfiles:
    measurement.append(pickle.load(open(out,"rb")))
o.calc_cost(measurement)
'''


