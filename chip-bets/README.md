# 2017ChipBets

Your have 2 bags containing 200 chips each.

There are red and white chips distributed between the two bags:

    White dominant bag: 160 white chips and 40 red chips
    Red dominant bag:   120   red chips and 80 white chips
    
You have a bank account of $1000.

The object is to guess which bag chips are being selected from and to bet
up to 100% of your bank on the guess. E.g. you guess bag "wd" and bet $50.
If you are correct, you get $50; if you are wrong, you lose $50.

You will get 100 bets. At the beginning of each session, a bag
is selected at random. You can buy chips from that bag for $9.30
each using your bank account.  Chips are purchased 1 at a time and
you can switch from buying chips to betting at any time.

Chips are not replaced in the bags at any time. If you buy all of the chips from
any one bag, your session(s) is ended.

Highest bank account at the end of 100 bets wins.

Your strategy must be coded in the file "strategy.py" with the function
"bet_strategy" with the signature:

def bet_strategy(session, bank, chip_color, session_history, **kwargs)
    
    session is an int [0, 99] enumerating your bets
    bank is your current bank balance
    chip_color is the color of a chip if your last move was a "buy" ["w","r"]
    session_history is the list of all of your moves during the current session including
         the color of chips you bought ["w","r","u"]

Your function returns the tuple defining a chip buy or a bet:

    move_choice="bet" or "buy
    bet = [1, bank]
    bet_bag_dominant_color = "wd" or "rd"

I have provided the simulation code that will be used to evaluate your
strategy at the end of the competition in "sim.py". You can change anything
here you like, but there probabaly isn't any need. To see your bank account
balance at the end off 100 bets, code your strategy in "strategy.py" and
run "sim.py".

If you plan to do a parameter search, use the kwargs.  See meta.py for an
example of how this is done.

The example code shows a strategy of random selection base on buying a
single chip before betting. Run this example to see how it works, then
do something better!

Good luck!

## Scoring the submissions

Submissions are saved as `strategy.py.<Name>` (e.g. `strategy.py.Jeff`).
`submissions.py` loads them by name, so there is no need to copy one over
`strategy.py` to score it:

    ./sim.py                 # score strategy.py
    ./sim.py Jeff Josh       # score strategy.py.Jeff and strategy.py.Josh
    ./sim.py --all           # score every submission
    ./meta.py Jeff           # parameter search on strategy.py.Jeff

## Comparison notebook

`compare-strategies.ipynb` scores every `strategy.py.<Name>` in this folder
and compares them:

* the ranked result at the official seed (14338), checked against
  `sim.get_bank`, and each bank over the 100 bets;
* the same contest on 500 seeds (one seed is a single draw of luck): median,
  10th/90th percentile, chance of losing money, how often each strategy
  finishes first, and a head-to-head table;
* how each strategy plays a session: chips bought, bet size as a share of
  the bank, and accuracy.

To include a new entry, save it as `strategy.py.<Name>` and rerun the
notebook. It needs numpy, pandas, matplotlib, scipy (for Josh's entry) and
jupyter, e.g.

    uv run --with jupyter,matplotlib,pandas,numpy,scipy jupyter lab

## AI strategies

All the original entries buy one chip and bet, so they guess the bag right
about 70% of the time and differ only in bet size. The chip price is fixed
at $9.30 while the bank grows, so once the bank is large extra chips are
almost free and make the guess much more reliable. The catch is that the
two bags hold only 400 chips for all 100 sessions, and emptying a bag ends
a session with no bet. AI1 and AI2 buy several chips per session and bet
the Kelly fraction 2p - 1 of the bank, where p is the probability that the
chosen bag is the right one.

**AI1** (`strategy.py.AI1`) keeps no memory between sessions. It buys
chips until it is 95% sure of the bag, at most 3 per session; 3 per
session is at most 300 in total, so a bag can never run out.

**AI2** (`strategy.py.AI2`) keeps memory between sessions:

* whether each bet won tells it which bag that session used, so it tracks
  exactly what is left in each bag and computes exact probabilities;
* it spreads the remaining chips evenly over the remaining sessions, with
  a safety margin so a bag does not run out;
* it buys another chip only when that is expected to grow the bank by more
  than the chip costs;
* it bets half Kelly for the first 15 sessions, so an early loss while the
  bank is small doesn't sink the contest.

Results from the notebook (settings tuned on seeds 1001-1200, evaluated on
seeds 1-500):

| Strategy  | Official seed | Median (500 seeds) | Loses money | Finishes first |
|-----------|---------------|--------------------|-------------|----------------|
| AI2       | $2.0 x 10^14  | $7.4 x 10^13       | 2.2%        | 85%            |
| AI1       | $1.3 x 10^15  | $2.5 x 10^12       | 10.4%       | 14%            |
| FirstChip | $15.0M        | $3.1M              | 7.2%        | 0.2%           |
| Jeff      | $72.5k        | $43k               | 0.4%        | 1.2%           |
| Josh      | $290          | $270               | 100%        | 0.2%           |
| Random    | $0.40         | $5                 | 93%         | 0%             |

AI2 beats AI1 on 85% of seeds, though AI1 happens to finish ahead at the
official seed. Neither ever emptied a bag in the 500 runs.

AI2's memory between sessions isn't forbidden by the rules above, but it
is something the organizer may want to rule on. AI1 needs no memory.
