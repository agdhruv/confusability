# Post-training experiments

GSM8k noise experiment showed that base models maintain capabilities rapidly with noisy data, and instruction-tuned models lose capabilities more rapidly.

In this directory, I tried to post-train a base model to see at what step of the post-training pipeline the model starts to lose capabilities. Unfortunately, I couldn't recreate the behavior of full instruction-tuned models (likely because I can't train at that scale anyway).