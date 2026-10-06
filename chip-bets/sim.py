#!/usr/bin/env python3
""" This code calculates bank based on your strategy. 

    You can do anything you like with this file! But I will use
    the original version to evaluate your strategy so change it with care. 
    
    I will use a dfferent seed_value (but the same one for all of the
    submissions) to evaluate the strategies.    

    Usage:
        ./sim.py                 score strategy.py
        ./sim.py Jeff Josh       score strategy.py.Jeff and strategy.py.Josh
        ./sim.py --all           score every strategy.py.<Name> submission
"""
from strategy import *
from submissions import list_submissions, load_strategy

import argparse
import random

def get_bank(strategy_fn=None, **kwargs):
    # strategy_fn: a bet_strategy function to score; default is strategy.py's
    if strategy_fn is None:
        strategy_fn = bet_strategy

    logging.debug("#"*60)
    logging.debug("{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
    logging.debug("#"*60)

    ##############################
    seed_value = 14338
    ##############################
    random.seed(a=seed_value)
    logging.debug("seed={}".format(seed_value))

    # Bags with chips as defined in problem statement
    # Note: If you want to cheat, just pass these 
    # variables to your strategy function! :)
    white_dominant = ["w"]*n_wd_w + ["r"]*n_wd_r 
    red_dominant =   ["r"]*n_rd_r + ["w"]*n_rd_w
    random.shuffle(white_dominant)
    random.shuffle(red_dominant)
    bags = {"wd": white_dominant, "rd": red_dominant}

    # parameters
    n_sessions = 100
    bank = 1000
    logging.debug("n_sessions={} bank=${:,.2f} chip_cost=${:,.2f}".format(n_sessions, bank, chip_cost))

    # sequence - predetermined so use of random doesn't change sequence
    # based on strategy
    # Python3.6:
    # bag_keys = random.choices(["rd","wd"], k=n_sessions)
    bag_keys = []
    for _ in range(n_sessions):
        bag_keys.append(random.choice(["rd", "wd"]))

    # betting sessions
    n_session = 0
    while bank >= 0 and n_session < n_sessions:
        bag_key = bag_keys[n_session]
        logging.debug("{} bank=${} bag_key={}".format(n_session, bank, bag_key))

        chip_color = "u"
        session_history = []
        
        # buy or bet?
        cont = True
        while cont:
            ###########################################################
            move_choice, bet, bet_bag = strategy_fn(n_session, bank, chip_color, session_history, **kwargs)
            logging.debug("strategy: {} bank=${} ({},{},{})".format(n_session, bank, move_choice, bet, bet_bag))
            ###########################################################
            if move_choice.lower() == "bet":
                chip_color = 'u'
                # only integer bets
                bet = int(bet)
                # minimum bet is $1
                if bet < 1:
                    bet = 1
                if bank > bet:
                    if bet_bag == bag_key:
                        bank += bet
                    else:
                        bank -= bet
                    #
                    cont = False
                else: 
                    logging.error("{}-{} bank=${} {}".format(n_session, move_choice, bank,"Error, bank < bet"))
                    break

            elif move_choice.lower() == "buy":
                try:
                    # policy is that empty bag errors out
                    chip_color = bags[bag_key].pop()
                except IndexError:
                    logging.error("{}-{} Bag empty!".format(n_session, move_choice))
                    break
                #
                if bank > chip_cost:
                    bank -= chip_cost
                else:
                    logging.error("{}-{} bank=${} {}".format(n_session, move_choice, bank,"Error, bank < 1"))
                    break

            else:
                logging.error("{} bank=${}".format(n_session, move_choice, bank,"Not a valid move choice!"))
                break
            session_history.append((move_choice, bet, bet_bag, chip_color))
            logging.debug("history: {}".format(session_history))
        n_session +=1 

    logging.debug("{} bank=${} ***FINAL***".format(n_session, bank))
    return bank

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score chip-bet strategies.")
    parser.add_argument("submissions", nargs="*",
                        help="submission names (strategy.py.<Name>) or file paths; default strategy.py")
    parser.add_argument("--all", action="store_true", help="score every strategy.py.<Name> submission")
    args = parser.parse_args()

    names = list(list_submissions()) if args.all else args.submissions
    if not names:
        print("Final Bank=${:,.2f}".format(get_bank()))
    for name in names:
        print("{:>12} Final Bank=${:,.2f}".format(name, get_bank(load_strategy(name))))



        

