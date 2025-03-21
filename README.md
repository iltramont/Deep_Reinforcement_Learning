# DLA Exam's Project by Federico Lancini, Alessandro Maggioni, and Luca Tramonti.
The aim of this project is to create and compare three different models to solve the problem of the double inverted pendulum (in the model of the environment that is made by OpenAI in the Gym library).

The authors created this code with the idea of the following execution order:

1. `environment_inspection.ipynb` shows the main features of the environment and presents a short description.

2. (If you do not re-train the models and are using the models that are already available in the `model` folder, you can skip this point) `DDPG_training.ipynb`, `DQL_training.ipynb`, and `PPO_training.ipynb` respectively train the different models and test their performance over time.

3. In `testing.ipynb`, the authors test the models and compare them.

4. In `DDPG_testing.ipynb`, we test the agent with different checkpoints and different `malfunction_probability` values. The `malfunction_probability` represents the probability of the environment having a malfunction in the sense that the force that is really applied to the cart is less than the force that the agent applies.