
    # for i in range(1):
    #     time0 = time.time()
    #     print(f"#{i + 1} training progress")
    #     train(env, 200)
    #     time1 = time.time()
    #     print(f"Training time: {time1 - time0} seconds")
    #     print ("Win rate: ", env.win_count ,"/", env.win_count + env.dead_count, f"({env.get_win_rate()})")
    #     [profit, loss] = env.get_cumulative_profit_loss_ratio()
    #     print("Profit Loss Ratio: ",f"{profit} : {loss}" )
    #     print ("Final profit rate: ", env.get_profit_rate())