import argparse
import time

from typing import Tuple
import aiohttp
import asyncio
import time
import json

async def vllm_inference_call_server(prompt, in_num, out_num, sampled_in_num, sampled_out_num, sleep_time, config, logger, event_id):
    await asyncio.sleep(sleep_time)
    # if event_id >= 140 and event_id <= 150:
    #     return
    # if event_id >= 180 and event_id <= 185:
    #     return
    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
    print(f"[INFO] Start {event_id}, after sleep: {sleep_time}")
    async with aiohttp.ClientSession(timeout=timeout) as session:
        generation_input = {
            "prompt": prompt,
            "stream": config.server_config['stream'],
            "ignore_eos": True, # TODO
            # "ignore_eos": True,
            "max_tokens": int(out_num),
            # "min_tokens": int(out_num) - 64,
            "temperature": config.server_config['temperature'],
        }
        first_chunk_time = 0
        start_time = time.perf_counter()
        # try:
        async with session.post(
            f"http://{config.server_config['host']}:{config.server_config['port']}/generate", json=generation_input
        ) as resp:
            if resp.status != 200:
                print(f"Error: {resp.status} {resp.reason}")
                print(await resp.text())
                return None, None, None

            if config.server_config['stream']:
                buffer = b""
                json_str = None
                first_chunk_received = False
                async for chunk in resp.content.iter_any():
                    buffer += chunk

                    # If this is the first chunk, record the time taken
                    if not first_chunk_received:
                        first_chunk_time = time.perf_counter() - start_time
                        first_chunk_received = True

                    while b"\n" in buffer:  # Split by null character
                        json_str, buffer = buffer.split(b"\n", 1)
                
                output = json.loads(json_str.decode("utf-8"))  # Decode JSON

            else:
                output = await resp.json()
            
            end_time = time.perf_counter()
            total_chunk_time = end_time - start_time

                # should counter the output token length after gather all the outputs
        # except:     # TODO
        #     with open(logger.log_path, "a") as f:
        #         f.write(f"{event_id} client failed \n")
        #     return
    
    logger.tick_end(event_id, time.perf_counter())

    save_query_json = {"event_id":event_id, "out_len":len(output["text"][0]), "out_len_expected": int(out_num), "in_len":int(in_num),"sampled_in_num": int(sampled_in_num), "sampled_out_len":int(sampled_out_num), "first_chunk_time":first_chunk_time, "total_chunk_time":total_chunk_time, "record_time":time.perf_counter()}
    output_save_query_json = {"output": output["text"][0]}

    with open(logger.log_path, "a") as f:
        f.write("\n")
        json.dump(save_query_json, f)
        json.dump(output_save_query_json, f)