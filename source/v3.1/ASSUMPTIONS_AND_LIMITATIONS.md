# Assumptions and Limitations

* The 20-year series is statistical augmentation, not 20 observed years.
* This is an hourly energy/supervisory model, not voltage, reactive-power,
converter-switching, protection, or electromagnetic-transient simulation.
* The configurable shared BESS enforces SoC, power, efficiency, and balance.
* Degradation is charged exactly once operationally; replacement NPV is reported separately.
* MPC uses a 24-h horizon and replans every six hours to make full-year studies tractable.
* PER and uniform replay share architecture, windows, seeds, and hyperparameters.
* Three seeds are a matched sensitivity analysis, not proof that RL always wins.
* Heterogeneous pricing is selected from training data per household and fixed for test T.
* Daily Mid-C is historical; the intraday residential RTP component is modeled.
* Outages are scenario simulations; SAIDI/SAIFI/CAIFI are not historical BC Hydro claims.
* Causal seasonal-naive forecasts are integrated; LSTM files remain as references.

