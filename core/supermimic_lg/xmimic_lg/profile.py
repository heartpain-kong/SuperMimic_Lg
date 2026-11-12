"""Explicit, overrideable defaults for the memory-sized LgDog XMimic port."""


DEFAULTS = dict(update_to_data_ratio=.4, horizon=32, wm_horizon=8, wm_updates=5,
    policy_updates=1, warmup_iterations=1, warmup_steps=64, wm_batch_size=512,
    policy_batch_size=256, critic_epochs=5, critic_minibatches=4, wm_lr=4e-4,
    actor_lr=4e-4, critic_lr=4e-4, actor_beta2=.95, actor_grad_value=.1,
    normalize_actor_loss=False, critic_hidden=(256, 128), policy_ema_decay=.995,
    wm_mirror_fraction=0., lr_decay_iterations=250000, policy_lr_mult=4.,
    policy_lr_decay_start=1400, policy_lr_decay_length=300)


def resolve_profile(namespace, argv):
    if namespace.algorithm != "xmimic":
        return namespace
    options = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    for key, value in DEFAULTS.items():
        flag = "--"+key.replace("_", "-")
        if flag in options or (key == "normalize_actor_loss" and "--no-normalize-actor-loss" in options):
            continue
        # The legacy global batch is also an explicit batch override.
        if key in ("wm_batch_size", "policy_batch_size") and "--batch-size" in options:
            continue
        setattr(namespace, key, value)
    return namespace
