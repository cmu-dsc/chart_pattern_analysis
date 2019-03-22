import sys
sys.path.append('/export/home/math/trose/python_utils')
import math_utils

def see_if_buy_executes(dct, buy_today, bought_today, tib, i, h=2.0, commission=0.005):
    '''
    dct: dict
        Dictionary of symbols and their data for today
    buy_today: list of lists
        Each list in buy_today is a stock that matched the pattern and details about it.
        This trade may or may not have been executed yet. If it has been estimated to
        have executed (in simulations) then it will appear in bought_today as well.
        It has form: 
        [sym, target_buy_price, hop_pct, stop_pct, method, buy_i]
    bought_today: list of lists
        Each list in bought_today is a stock that matched the pattern and details about it.
        This trade has been estimated to have executed (in simulations).
        It has form:
        sym, target_buy_price, cost_basis, max_when_buy, dop, pop_pct, hop_pct, stop_pct, vol, method, buy_i
    tib: number
        tib is the number of seconds that is pretty safe as to how long it takes 
        to submit a trade thru IBgateway. It will take at least this long for a 
        buy signal to be put on tape, so thus we'll need to start counting volumes
        added to tape after this point to see if the order will go through.
    h: number
        In estimating whether a buy (obtaining the position) will go through,
        (for use in simulations) you'll likely get your target buy price if
        h * vol gets put on tape - after the point in time where the
        buy target was reached - with these volumes having corresponding Mark prices 
        at or below the target buy price. h can be better determined through actual 
        trading. Likely bounds for h are 1 <= h <= 3? where h can be more than 1 
        due to other people getting their trades to execute before you (because they
        placed their trade order before you).
    
    Return: list of lists
        Any trades that are estimated to have been executed will be appended to bought_today
    Purpose: In estimating whether a buy (obtaining the position) will go through,
        (for use in simulations) you'll likely get your target buy price if
        h * vol gets put on tape - after the point in time where the
        buy target was reached - with these volumes having corresponding Mark prices
        at or below the target buy price. This is an approximation because data is only
        collected once per second and it is unknown how many trades and for what volumes
        were posted before your order by other traders.
    '''

    bought_list_syms = [bought_list[0] for bought_list in bought_today]

    for buy_list in buy_today:
        if buy_list[0] in bought_list_syms:
            continue
        sym, target_buy_price, cost_basis, max_when_buy, dop, pop_pct, hop_pct, stop_pct, vol_request, method, buy_i = buy_list
        if i <= buy_i + tib + 1:
            continue
        mark_list = dct[sym][:i,0][buy_i + tib + 1:]
        vol_list = dct[sym][:i,1][buy_i + tib + 1:]
        vol_at_or_below_target = int(sum([vol for j,vol in enumerate(vol_list) if mark_list[j] <= target_buy_price]))
        if vol_at_or_below_target >= int(h * vol_request):
            bought_i = i - 1
            bought_today.append(buy_list + [bought_i])
    return bought_today

def get_net(buy_price, sell_price, vol_requested, commission=0.005):
    '''
    buy_price: float
        price you bought the stock for
    sell_price: float
        price you bought the stock for
    vol_requested: number
        size of the position
    commission: float
        price per share in commission

    return: float
        net value in dollars of the result of the trade
    '''
    vol_requested = float(vol_requested)
    commission_total = 2.0 * commission * vol_requested
    net = (buy_price - sell_price) * vol_requested - commission_total
    return round(net, 3)

def see_if_sell_executes(dct, sold_today, bought_today, buying_power_available, i, h=2.0):
    '''
    dct: dict
        Dictionary of symbols and their data for today
    sold_today: list of lists
        Each list in sold_today is a stock that was sold already.
        It has form:
        [sym, target_buy_price, cost_basis, max_when_buy, dop, pop_pct, hop_pct, stop_pct, vol_request, method, buy_i, bought_i, sell_i, net]
    bought_today: list of lists
        Each list in bought_today is a stock that matched the pattern and details about it.
        This trade has been estimated to have executed (in simulations).
        It has form:
        [sym, target_buy_price, max_when_buy, dop, pop_pct, hop_pct, stop_pct, vol_request, method, buy_i, bought_i]
    buying_power_available: float
        current available buying power
    h: number
        In estimating whether a sell (closing the position) will go through,
        (for use in simulations) you'll likely get your target sell price if
        h * vol gets put on tape - after the point in time where the
        the postion was obtained was reached - with these volumes having corresponding Mark prices
        at or above the target sell price. h can be better determined through actual
        trading. Likely bounds for h are 1 <= h <= 3? where h can be more than 1
        due to other people getting their trades to execute before you (because they
        placed their trade order before you).

    Return: list of lists
        Any trades that are estimated to have been executed will be appended to sold_today
    Purpose: In estimating whether a buy (obtaining the position) will go through,
        (for use in simulations) you'll likely get your target sell price if
        h * vol gets put on tape - after the point in time where the
        sell target was reached - with these volumes having corresponding Mark prices
        at or above the target buy price. This is an approximation because data is only
        collected once per second and it is unknown how many trades and for what volumes
        were posted before your order by other traders.
    '''

    sold_list_syms = [sold_list[0] for sold_list in sold_today]

    for bought_list in bought_today:
        if bought_list[0] in sold_list_syms:
            continue
        sym, target_buy_price, cost_basis, max_when_buy, dop, pop_pct, hop_pct, stop_pct, vol_request, method, buy_i, bought_i = bought_list
        
        if i <= bought_i + 1:
            continue
        if method == 3:
            target_sell_price = round(target_buy_price * hop_pct, 3)
            target_stop_price = round(target_buy_price * stop_pct, 3)
        mark_list = dct[sym][:i,0][bought_i + 1:]
        vol_list = dct[sym][:i,1][bought_i + 1:]
        vol_at_or_above_target = int(sum([vol for j,vol in enumerate(vol_list) if mark_list[j] >= target_sell_price]))
        if vol_at_or_above_target >= int(h * vol_request):
            sold_i = i - 1
            net = get_net(target_buy_price, target_sell_price, vol_request, commission=0.005)
            sold_today.append(bought_list + [sold_i, net])
            buying_power_available += round(vol_request * target_sell_price, 3)
        else:
            vol_at_or_below_target = int(sum([vol for j,vol in enumerate(vol_list) if mark_list[j] <= target_stop_price]))
            if vol_at_or_below_target >= int(h * vol_request):
                sold_i = i - 1
                net = get_net(target_buy_price, target_stop_price, vol_request, commission=0.005)
                sold_today.append(bought_list + [sold_i, net])
                buying_power_available += round(vol_request * target_stop_price, 3)
    return sold_today, buying_power_available

def close_open_positions(dct, sold_today, bought_today, i):
    '''
    dct: dict
        Dictionary of symbols and their data for today
    sold_today: list of lists
        Each list in sold_today is a stock that was sold already.
        It has form:
        [sym, target_buy_price, cost_basis, max_when_buy, dop, pop_pct, hop_pct, stop_pct, vol_request, method, buy_i, bought_i, sell_i, net]
    bought_today: list of lists
        Each list in bought_today is a stock that matched the pattern and details about it.
        This trade has been estimated to have executed (in simulations).
        It has form:
        [sym, target_buy_price, max_when_buy, dop, pop_pct, hop_pct, stop_pct, vol_request, method, buy_i, bought_i]

    return: list of lists
        updated sold_today
    Purpose: At the end of the day (15:45:00), close all remaining positions with market orders
        so we can clear the slate for tomorrow.
    '''
    sold_list_syms = [sold_list[0] for sold_list in sold_today]
    for bought_list in bought_today:
        if bought_list[0] in sold_list_syms:
            continue
        sym, target_buy_price, cost_basis, max_when_buy, dop, pop_pct, hop_pct, stop_pct, vol_request, method, buy_i, bought_i = bought_list

        sold_i = i - 1
        net = get_net(target_buy_price, dct[sym][:i,0][-1], vol_request, commission=0.005)
        sold_today.append(bought_list + [sold_i, round(net, 3)])
    return sold_today
