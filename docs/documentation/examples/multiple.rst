Example Codes with Multiple Logged Dataset and Behavior Policies
==========

Here, we show example codes for conducting OPE and OPS with multiple logged dataset.

.. seealso::

    For preparation, please also refer to the following pages about the case with a single logged dataset:

    * :doc:`Example codes for data collection and Offline RL </documentation/examples/data_collection_and_opl>`
    * :doc:`Example codes for basic OPE </documentation/examples/basic_ope>`
    * :doc:`Example codes for cumulative distribution OPE </documentation/examples/cumulative_dist_ope>`
    * :doc:`Example codes for OPS </documentation/examples/ops>`
    * :doc:`Example codes for assessing OPE and OPS </documentation/examples/assessments>`

Logged Dataset
~~~~~~~~~~
Here, we assume that an RL environment, behavior policies, and evaluation policies are given as follows.

* ``behavior_policy``: an instance of :class:`BaseHead` or a list of instance(s) of :class:`BaseHead` 
* ``evaluation_policies``: a list of instance(s) of :class:`BaseHead`
* ``env``: a gym environment (unecessary when using real-world datasets)

Then, we can collect multiple logged datasets with a single behavior policy as follows.

.. code-block:: python

    from scope_rl.dataset import SyntheticDataset
    
    # initialize dataset
    dataset = SyntheticDataset(
        env=env,
        max_episode_steps=env.step_per_episode,
    )
    # obtain logged dataset
    multiple_logged_datasets = dataset.obtain_episodes(
        behavior_policies=behavior_policies[0],  # a single behavior policy
        n_datasets=5,  # specify the number of dataset (i.e., number of different random seeeds)
        n_trajectories=10000, 
        random_state=random_state,
    )

Similarly, SCOPE-RL also collects multiple logged datasets with multiple behavior policies as follows.

.. code-block:: python

    multiple_logged_datasets = dataset.obtain_episodes(
        behavior_policies=behavior_policies,  # multiple behavior policies
        n_datasets=5,  # specify the number of dataset (i.e., number of different random seeeds) for each behavior policy
        n_trajectories=10000, 
        random_state=random_state,
    )

The multiple logged datasets are returned as an instance of :class:`MultipleLoggedDataset`. 
Note that, we can also manually create multiple logged datasets as follows.

.. code-block:: python

    from scope_rl.utils import MultipleLoggedDataset

    multiple_logged_dataset = MultipleLoggedDataset(
        action_type="discrete",
        path="logged_dataset/",  # specify the path to the dataset
    )

    for behavior_policy in behavior_policies:
        single_logged_dataset = dataset.obtain_episodes(
            behavior_policies=behavior_policy,  # a single behavior policy
            n_trajectories=10000,
            random_state=random_state,
        )

        # add a single_logged_dataset to multiple_logged_dataset
        multiple_logged_dataset.add(
            single_logged_dataset,
            behavior_policy_name=behavior_policy.name,
        )

Once you create the multiple logged datasets, each dataset is accessible via the following code.

.. code-block:: python

    single_logged_dataset = multiple_logged_dataset.get(
        behavior_policy_name=behavior_policies[0].name, dataset_id=0,
    )

:class:`MultipleLoggedDataset` also has the following properties.

.. code-block:: python

    # a list of the name of behavior policies
    multiple_logged_dataset.behavior_policy_names

    # a dictionary of the number of datasets for each behavior policy
    multiple_logged_dataset.n_datasets

Inputs
~~~~~~~~~~
