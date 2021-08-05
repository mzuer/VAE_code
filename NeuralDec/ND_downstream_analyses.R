fpath="ND_METABRIC_MRNA/vae_model_hd32_nLD1_cER_Status_300_100_15_dict.pt"

state_dict <- load_state_dict(fpath)
model <- Model()
model$load_state_dict(state_dict)state_dict <- load_state_dict(fpath)
model <- Model()
model$load_state_dict(state_dict)