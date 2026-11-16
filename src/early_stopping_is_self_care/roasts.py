"""The jokes. All original, all kind, all a bit nerdy.

Placeholders available to every template:

* ``{best_epoch}``   1-based epoch with the best monitored value
* ``{last_epoch}``   1-based epoch the run ended on
* ``{extra_epochs}`` e.g. "1 epoch" / "5 epochs" (last_epoch - best_epoch)
* ``{monitor}``      the monitored metric name, e.g. ``val_loss``
* ``{best}``         the best monitored value, pre-formatted
* ``{patience}``     the patience setting
"""

from __future__ import annotations

from typing import Tuple

# Safe whichever way the run ended: they never claim early stopping fired.
GENERAL: Tuple[str, ...] = (
    "Your {monitor} peaked at epoch {best_epoch}. Everything after was just "
    "memorising the training set's birthday.",
    "Epoch {best_epoch} was the one. The next {extra_epochs} were the model "
    "re-reading old messages from the training set at 2am.",
    "Best {monitor} at epoch {best_epoch}. Since then the model has been learning "
    "the noise, and the noise has learned nothing in return.",
    "Validation gently said \"it's not you, it's your variance\" at epoch {best_epoch}.",
    "At epoch {best_epoch} your model generalised. After that it just got very good "
    "at one specific dataset, like a pub quiz regular who only knows one quiz.",
    "The training loss kept going down. The validation loss quietly moved out "
    "at epoch {best_epoch} and took the houseplants.",
    "Best {monitor}: {best}. Anything after epoch {best_epoch} was gradient descent "
    "into mild, cosy madness.",
    "{extra_epochs} more of training, zero more of learning. That's not a run, "
    "that's a treadmill with a loss curve.",
    "Bias and variance walked into epoch {best_epoch} as friends. Variance left alone.",
    "Somewhere after epoch {best_epoch} the model stopped finding patterns and "
    "started seeing faces in the clouds.",
    "Training past the best epoch is like re-reading your exam answers until you "
    "start memorising the page numbers.",
    "Congratulations: the model now knows the training set's favourite colour. "
    "Epoch {best_epoch} would like a quiet word.",
    "The best weights were at epoch {best_epoch}. Take them, drink some water, "
    "close the laptop.",
    "That's {extra_epochs} of the model nodding along to the training set "
    "without really listening.",
    "Generalisation peaked at epoch {best_epoch}. Since then: lookup-table cosplay.",
    "Your learning rate did nothing wrong. Your epoch count just had big dreams.",
    "{monitor} bottomed out at epoch {best_epoch}. The rest was the model learning "
    "every mislabelled example by name.",
    "Your loss curve smiled at epoch {best_epoch}. Everything after is that smile "
    "slowly turning into a smirk.",
    "The model hit its best at epoch {best_epoch}. Peaking early is fine; most of "
    "us peaked at a hackathon.",
    "Post-epoch-{best_epoch} training is basically fan fiction about the training set.",
    "Your validation set has been politely waving a white flag since epoch {best_epoch}.",
    "Epoch {best_epoch}: generalisation. Epoch {last_epoch}: a very confident lookup "
    "table with opinions.",
)

# Only used when early stopping (or something else) actually stopped the run.
STOPPED: Tuple[str, ...] = (
    "Early stopping fired at epoch {last_epoch}. It's not quitting, it's setting boundaries.",
    "Patience is a virtue. patience={patience} is a hyperparameter. Both just ran out "
    "at epoch {last_epoch}.",
    "Early stopping is self-care. Your model clocked off at epoch {last_epoch}, and so can you.",
    "Stopped at epoch {last_epoch}: the computational equivalent of leaving the party "
    "while it's still fun.",
    "No improvement in {monitor} for {extra_epochs}. The optimiser wasn't stuck, it was "
    "meditating. Early stopping respected that.",
    "Early stopping noticed what validation had been hinting since epoch {best_epoch}. "
    "Healthy communication, finally.",
    "Epochs {best_epoch} to {last_epoch}: a montage of the model rehearsing answers to "
    "questions nobody will ask. Mercifully, early stopping called cut.",
)

# Used when the run ended on its own but after its best epoch.
PASSED_PEAK: Tuple[str, ...] = (
    "Nobody stopped this run, so it kept going for {extra_epochs} past its best "
    "(epoch {best_epoch}). This callback can suggest rest; it can't confiscate your GPU.",
    "The run ended at epoch {last_epoch}, but its best day was epoch {best_epoch}. "
    "Consider EarlyStopping: it's a bedtime, but for gradients.",
    "max_epochs reached. Best {monitor} was back at epoch {best_epoch}. The last "
    "{extra_epochs} were a very expensive way to keep the room warm.",
)

FALLBACK = "Training stopped. Rest is a regulariser."
