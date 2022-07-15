import logging
from random import Random

import multineat
from genotype import random as random_genotype
# from optimizer import Optimizer
from optimizer_rugged import Optimizer

from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import ProcessIdGen
from direct_tree.direct_tree_config import DirectTreeGenotypeConfig


async def main() -> None:
    # number of initial mutations for body and brain CPPNWIN networks
    NUM_INITIAL_MUTATIONS = 20

    SIMULATION_TIME = 10
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 20

    POPULATION_SIZE = 50
    OFFSPRING_SIZE = 50
    NUM_GENERATIONS = 50

    MAX_PARTS = 10
    N_PARTS_MU = 5
    N_PARTS_SIGMA = 3
    config = DirectTreeGenotypeConfig()


    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    logging.info(f"Starting optimization")

    # database
    database = open_async_database_sqlite("./tree-database")

    # process id generator
    process_id_gen = ProcessIdGen()

    for experiment in range(5):
        # random number generator
        rng = Random()
        rng.seed(experiment)

        # multineat innovation databases
        innov_db_brain = multineat.InnovationDatabase()

        initial_population = [
            random_genotype(innov_db_brain=innov_db_brain,
                            rng=rng,
                            num_initial_mutations=NUM_INITIAL_MUTATIONS,
                            max_parts=MAX_PARTS,
                            n_parts_mu=N_PARTS_MU,
                            n_parts_sigma=N_PARTS_SIGMA,
                            config=config)
            for _ in range(POPULATION_SIZE)
        ]

        process_id = process_id_gen.gen()
        # print(process_id_gen.get_state())
        maybe_optimizer = await Optimizer.from_database(
            database=database,
            process_id=process_id,
            innov_db_brain=innov_db_brain,
            rng=rng,
            process_id_gen=process_id_gen,
            treeconfig=config
        )
        if maybe_optimizer is not None:
            optimizer = maybe_optimizer
        else:
            optimizer = await Optimizer.new(
                database=database,
                process_id=process_id,
                initial_population=initial_population,
                rng=rng,
                process_id_gen=process_id_gen,
                innov_db_brain=innov_db_brain,
                simulation_time=SIMULATION_TIME,
                sampling_frequency=SAMPLING_FREQUENCY,
                control_frequency=CONTROL_FREQUENCY,
                num_generations=NUM_GENERATIONS,
                offspring_size=OFFSPRING_SIZE,
                treeconfig=config
            )

        logging.info("Starting optimization process..")

        await optimizer.run()

        logging.info(f"Finished optimizing.")

if __name__ == "__main__":
    import asyncio
    if True:
        import shutil
        import os
            # remove this when you want to keep data in database file
        dir_path = '/home/ibilgin/Desktop/tree-database'

        try:
            shutil.rmtree(dir_path)
            print(" I removed database")

        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))

        files = '/home/ibilgin/Desktop/ilayda-tree/morphologies'
        try:
            shutil.rmtree(files)
            os.mkdir(files)

        except OSError as e:
            print("Error: %s : %s" % (files, e.strerror))

    asyncio.run(main())
