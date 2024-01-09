import glob
import time
import numpy as np
from hashlib import sha256
from BCModel import *
import pickle
import json
import ast


def addNewUpdate(blockchain):
    life = 1
    
    while life > 0:
        update_list = glob.glob("update/*")
        if update_list:
            for f in update_list:
                new_record = {}
                new_record["userID"] = str(f.split('/')[1].split('.')[0])
                new_record["update"] = str(np.load(f))
                new_record["timestamp"] = time.time()#prevent users from creating fake time
                
                blockchain.add_new_transaction(new_record)
                #creat record
                life -= 1
        else:
            print("sleep")
            time.sleep(5)
            continue
    
def consensus():
    """
    Our naive consnsus algorithm. If a longer valid chain is
    found, our chain is replaced with it.
    """
    global blockchain

    longest_chain = None
    current_len = len(blockchain.chain)

    for node in peers:
        response = requests.get('{}chain'.format(node))
        length = response.json()['length']
        chain = response.json()['chain']
        if length > current_len and blockchain.check_chain_validity(chain):
            current_len = length
            longest_chain = chain

    if longest_chain:
        blockchain = longest_chain
        return True

    return False


def announce_new_block(block,peers):
    """
    A function to announce to the network once a block has been mined.
    Other blocks can simply verify the proof of work and add it to their
    respective chains.
    """
    for peer in peers:
        url = "{}add_block".format(peer)
        headers = {'Content-Type': "application/json"}
        requests.post(url,
                      data=json.dumps(block.__dict__, sort_keys=True),
                      headers=headers)    
 
    
def create_chain_from_dump(chain_dump):
    generated_blockchain = Blockchain()
    generated_blockchain.create_genesis_block()
    for idx, block_data in enumerate(chain_dump):
        if idx == 0:
            continue  # skip genesis block
        block = Block(block_data["index"],
                      block_data["transactions"],
                      block_data["timestamp"],
                      block_data["previous_hash"],
                      block_data["nonce"])
        proof = block_data['hash']
        added = generated_blockchain.add_block(block, proof)
        if not added:
            raise Exception("The chain dump is tampered!!")
    return generated_blockchain
