import math
import pickle
from random import Random
from typing import List, Tuple

import multineat
import sqlalchemy
import numpy
import pandas
from genotype import Genotype, GenotypeSerializer, crossover, develop, mutate
from pyrr import Quaternion, Vector3
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

import revolve2.core.optimization.ea.generic_ea.population_management as population_management
import revolve2.core.optimization.ea.generic_ea.selection as selection
from direct_tree.direct_tree_config import DirectTreeGenotypeConfig
#from direct_tree.direct_tree_utils import bfs_iterate_modules, recursive_iterate_modules

from revolve2.actor_controller import ActorController
from revolve2.core.database import IncompatibleError
from revolve2.core.database.serializers import FloatSerializer
from revolve2.core.optimization import ProcessIdGen
from revolve2.core.optimization.ea.generic_ea import EAOptimizer
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    Environment,
    PosedActor,
    Runner,
)
from revolve2.runners.isaacgym._local_runner import LocalRunner


class Optimizer(EAOptimizer[Genotype, float]):
    _process_id: int

    _runner: Runner

    _controllers: List[ActorController]

    _innov_db_brain: multineat.InnovationDatabase

    _rng: Random

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_generations: int
    _treeconfig: DirectTreeGenotypeConfig

    async def ainit_new(
            # type: ignore # TODO for now ignoring mypy complaint about LSP problem, override parent's ainit
            self,
            database: AsyncEngine,
            session: AsyncSession,
            process_id: int,
            process_id_gen: ProcessIdGen,
            initial_population: List[Genotype],
            rng: Random,
            innov_db_brain: multineat.InnovationDatabase,
            simulation_time: int,
            sampling_frequency: float,
            control_frequency: float,
            num_generations: int,
            offspring_size: int,
            treeconfig: DirectTreeGenotypeConfig
    ) -> None:
        await super().ainit_new(
            database=database,
            session=session,
            process_id=process_id,
            process_id_gen=process_id_gen,
            genotype_type=Genotype,
            genotype_serializer=GenotypeSerializer,
            fitness_type=float,
            fitness_serializer=FloatSerializer,
            offspring_size=offspring_size,
            initial_population=initial_population,
        )

        self._process_id = process_id
        self._init_runner()
        self._innov_db_brain = innov_db_brain
        self._rng = rng
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations
        self._treeconfig = treeconfig

        # create database structure if not exists
        # TODO this works but there is probably a better way
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

        # save to database
        self._on_generation_checkpoint(session)
        self.morphologies = []
        self.data_analysis = {'gen': [], 'module_count': [], 'ratio_w/l': [], 'joints_count': [], 'coverage': [],
                              'symmetry_score': [], 'module_symmetry': []}

    async def ainit_from_database(  # type: ignore # see comment at ainit_new
            self,
            database: AsyncEngine,
            session: AsyncSession,
            process_id: int,
            process_id_gen: ProcessIdGen,
            rng: Random,
            innov_db_brain: multineat.InnovationDatabase,
            treeconfig: DirectTreeGenotypeConfig
    ) -> bool:
        if not await super().ainit_from_database(
                database=database,
                session=session,
                process_id=process_id,
                process_id_gen=process_id_gen,
                genotype_type=Genotype,
                genotype_serializer=GenotypeSerializer,
                fitness_type=float,
                fitness_serializer=FloatSerializer,
        ):
            return False

        self._process_id = process_id
        self._init_runner()
        self._treeconfig = treeconfig

        opt_row = (
            (
                await session.execute(
                    select(DbOptimizerState)
                        .filter(DbOptimizerState.process_id == process_id)
                        .order_by(DbOptimizerState.generation_index.desc())
                )
            )
                .scalars()
                .first()
        )

        # if this happens something is wrong with the database
        if opt_row is None:
            raise IncompatibleError

        self._simulation_time = opt_row.simulation_time
        self._sampling_frequency = opt_row.sampling_frequency
        self._control_frequency = opt_row.control_frequency
        self._num_generations = opt_row.num_generations

        self._rng = rng
        self._rng.setstate(pickle.loads(opt_row.rng))

        self._innov_db_brain = innov_db_brain
        self._innov_db_brain.Deserialize(opt_row.innov_db_brain)

        return True

    def _init_runner(self) -> None:
        self._runner = LocalRunner(LocalRunner.SimParams(), headless=True)

    def _select_parents(
            self,
            population: List[Genotype],
            fitnesses: List[float],
            num_parent_groups: int,
    ) -> List[List[int]]:
        return [
            selection.multiple_unique(
                population,
                fitnesses,
                2,
                lambda _, fitnesses: selection.tournament(self._rng, fitnesses, k=10),
            )
            for _ in range(num_parent_groups)
        ]

    def _select_survivors(
            self,
            old_individuals: List[Genotype],
            old_fitnesses: List[float],
            new_individuals: List[Genotype],
            new_fitnesses: List[float],
            num_survivors: int,
    ) -> Tuple[List[int], List[int]]:
        assert len(old_individuals) == num_survivors

        return population_management.steady_state(
            old_individuals,
            old_fitnesses,
            new_individuals,
            new_fitnesses,
            lambda _, fitnesses: selection.tournament(self._rng, fitnesses, k=10),
        )

    def _must_do_next_gen(self) -> bool:
        return self.generation_index != self._num_generations

    def _crossover(self, parents: List[Genotype]) -> Genotype:
        assert len(parents) == 2
        return crossover(parents[0], parents[1], self._rng, self._treeconfig)

    def _mutate(self, genotype: Genotype) -> Genotype:
        return mutate(genotype, self._innov_db_brain, self._rng, self._treeconfig)

    def morphology_visualisation(self, fitness_values, max_parts):
        import matplotlib.pyplot as plt
        from matplotlib import colors
        import numpy as np

        N = (max_parts*2)
        max_count = max_parts-1
        # make an empty data set
        # fill in some fake data
        representations = []
        plot_width = int(np.ceil(np.sqrt(len(self.morphologies))))

        fitness_copy = fitness_values[:]
        for morphology in self.morphologies:
            data = np.ones((N, N)) * np.nan

            for module in morphology:
                coordinate = module[1]
                module_ref = module[0]
                #module_or = module[2]
                if 'Core' in str(module_ref):
                    data[int(numpy.abs(coordinate[1] - max_count)), int(coordinate[0] + max_count)] = 0
                if 'Brick' in str(module_ref):
                    data[int(numpy.abs(coordinate[1] - max_count)), int(coordinate[0] + max_count)] = 1
                if 'ActiveHinge' in str(module_ref):
                    #if module_or[2] == 0:
                    data[int(numpy.abs(coordinate[1] - max_count)), int(coordinate[0] + max_count)] = 2
                    # else:
                    #     data[numpy.abs(coordinate[1] - max_parts), coordinate[0] + max_parts] = 3
            representations.append(data)
        # fill the canvas with empty plots
        for n in range(0, plot_width ** 2 - len(representations)):
            representations.append(np.ones((N, N)) * np.nan)
            fitness_copy.append(-1.0)

        ordered = [x for x in sorted(zip(fitness_copy, representations), key=lambda x: x[0], reverse=True)]
        # print(f'data:{len(ordered)}')


        # make a figure + axes
        fig, ax = plt.subplots(plot_width, plot_width, tight_layout=True, figsize=(10, 10))

        # make color map with boundaries
        norm = colors.BoundaryNorm([0., 0.5, 1.5, 2.5], 3)
        my_cmap = colors.ListedColormap(['y', 'b', 'r'])

        # set the 'bad' values (nan) to be white and transparent
        my_cmap.set_bad(color='w', alpha=0)

        # draw the boxes
        index = 0
        for x in range(plot_width):
            for y in range(plot_width):
                ax[x][y].imshow(ordered[index][1], interpolation='none', cmap=my_cmap, extent=[0, N, 0, N],
                                zorder=0, norm=norm)
                if fitness_copy[index] > 0.0:
                    ax[x][y].set_title('Fitness value: {:.4e}'.format(ordered[index][0]),
                                       fontsize=(45 * (1 / plot_width)))

                # turn off the axis labels
                ax[x][y].axis('off')
                index += 1
        plt.savefig(f'/home/ibilgin/Desktop/ilayda-tree/morphologies/generation_{self.generation_index}')

    def morphology_analysis(self):
        module_counts = []
        for morphology in self.morphologies:
            # count the number of modules in the morphology
            module_count = len([x[1] for x in morphology])
            component_count = 0

            # intializing variables
            modules = [x[0] for x in morphology]
            coordinates = [x[1] for x in morphology]
            coordinates = [list(map(int, x)) for x in coordinates]
            xy_coordinates = [(int(x[0]), int(x[1])) for x in coordinates]
            x_coordinates = [int(x[0]) for x in coordinates]
            y_coordinates = [int(y[1]) for y in coordinates]
            # chain_length = [x[3] for x in morphology]
            width = max(x_coordinates) - min(x_coordinates) + 1
            length = max(y_coordinates) - min(y_coordinates) + 1
            hor_center = (max(x_coordinates) + min(x_coordinates)) / 2
            ver_center = (max(y_coordinates) + min(y_coordinates)) / 2
            square_coordinates = []
            sym_score = 0
            module_sym = 0

            # coverage calculations
            for x in range(min(x_coordinates), max(x_coordinates) + 1):
                for y in range(min(y_coordinates), max(y_coordinates) + 1):
                    square_coordinates.append((x, y))
            square_fill = width * length
            for c in square_coordinates:
                if c not in xy_coordinates:
                    square_fill -= 1
            coverage = square_fill / (width * length)

            #counting the number joints
            number_of_joints = 0
            for module in modules:
                if 'ActiveHinge' in str(module):
                    number_of_joints += 1

            # calculating the symmetry score of the morphology
            for coordinate in coordinates:
                x_difference = coordinate[0] - hor_center
                y_difference = coordinate[1] - ver_center
                mirror_xcoor = [int(hor_center - x_difference), coordinate[1], coordinate[2]]
                mirror_ycoor = [coordinate[0], int(ver_center - y_difference), coordinate[2]]
                if x_difference != 0:
                    component_count += 1
                    if mirror_xcoor in coordinates:
                        sym_score += 1
                        if morphology[coordinates.index(coordinate)][1] == morphology[coordinates.index(mirror_xcoor)][
                            1]:
                            module_sym += 1
                if y_difference != 0:
                    component_count += 1
                    if mirror_ycoor in coordinates:
                        sym_score += 1
                        if morphology[coordinates.index(coordinate)][1] == morphology[coordinates.index(mirror_ycoor)][
                            1]:
                            module_sym += 1

            if component_count == 0:
                sym_score = 1.0
                module_sym = 1.0
            else:
                sym_score = (sym_score / component_count)
                module_sym = (sym_score / component_count)

            # calculating the ratio between width and length depending on which is largest
            ratio = min(width, length) / max(width, length)

            # adding data to dictionary
            self.data_analysis['gen'].append(self.generation_index)
            self.data_analysis['module_count'].append(module_count)
            self.data_analysis['coverage'].append(coverage)
            self.data_analysis['joints_count'].append(number_of_joints)
            self.data_analysis['ratio_w/l'].append(ratio)
            self.data_analysis['symmetry_score'].append(sym_score)
            self.data_analysis['module_symmetry'].append(module_sym)
            # self.data_analysis['limb_count'].append(number_of_limbs)

        # transferring data to csv file in the morphologies folder
        df_analysis = pandas.DataFrame(self.data_analysis)
        # print(df_analysis.shape)
        if self.generation_index + 1 == self._num_generations:

            import os.path

            if os.path.exists('/home/ibilgin/Desktop/ilayda-tree/morphologies/analysis_data.csv'):
                print(" I added to csv")
                df_analysis.to_csv('/home/ibilgin/Desktop/ilayda-tree/morphologies/analysis_data.csv', index=True,
                                   header=False, mode='a')
            else:
                print('I created new csv')
                df_analysis.to_csv(path_or_buf='/home/ibilgin/Desktop/ilayda-tree/morphologies/analysis_data.csv')

    async def _evaluate_generation(
            self,
            genotypes: List[Genotype],
            database: AsyncEngine,
            process_id: int,
            process_id_gen: ProcessIdGen,
    ) -> List[float]:
        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
            control=self._control,
        )

        self._controllers = []
        self.morphologies = []
        assert all(not g.body.genotype._is_finalized for g in genotypes), "finalized"
        assert all([[g1.body.genotype is g2.body.genotype for g2 in genotypes[i:]] for i, g1 in enumerate(genotypes)]), "duplicates"
        for genotype in genotypes:
            developed, morphology, max_parts = develop(genotype)
            self.morphologies.append(morphology)
            actor, controller = developed.make_actor_and_controller()

            bounding_box = actor.calc_aabb()
            self._controllers.append(controller)
            env = Environment()
            env.actors.append(
                PosedActor(
                    actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                    [0.0 for _ in controller.get_dof_targets()],
                ),
            )
            batch.environments.append(env)

        states = await self._runner.run_batch(batch)

        flying_indices = self._calculate_fitness_z(states, len(genotypes))
        fitness_values = [
            self._calculate_fitness(
                states[0].envs[i].actor_states[0],
                states[-1].envs[i].actor_states[0],
            )
            for i in range(len(genotypes))
        ]

        for index in flying_indices:
            print(True)
            fitness_values[index] = -0.1

        print("Fitness values are below:")
        print(sorted(fitness_values), len(fitness_values))

        self.morphology_visualisation(fitness_values, max_parts)
        self.morphology_analysis()

        return fitness_values

    @staticmethod
    def _calculate_fitness_z(states: ActorState, num_of_individuals) -> float:
        init_z_value = 0
        # list to store maximum z value per individual; len should be equal to pop size
        final_maximum_z_values = []
        # store indices of flying individuals
        indices_individuals = []

        # for each individual find the maximum z_value during simulation and append it tp final_maximum_z_values
        for i in range(num_of_individuals):

            for j in range(len(states)):
                # get current z_value for given i individual
                new_z_value = states[j].envs[i].actor_states[0].position[2]
                if new_z_value > init_z_value:
                    init_z_value = new_z_value
                # reset init_z_value since we are moving to next individual
                if j == (len(states) - 1):
                    final_maximum_z_values.append(init_z_value)
                    # save robot index to remove later from population
                    if init_z_value > 0.6:
                        indices_individuals.append(i)
                    init_z_value = 0

        print("Maximum z values are below:")
        print(sorted(final_maximum_z_values), len(final_maximum_z_values))
        return indices_individuals

    def _control(self, dt: float, control: ActorControl) -> None:
        for control_i, controller in enumerate(self._controllers):
            controller.step(dt)
            control.set_dof_targets(control_i, 0, controller.get_dof_targets())

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:
        # TODO simulation can continue slightly passed the defined sim time.

        # distance traveled on the xy plane
        fitness = float(
            math.sqrt(
                (begin_state.position[0] - end_state.position[0]) ** 2
                + ((begin_state.position[1] - end_state.position[1]) ** 2)
            )
        )

        if math.isnan(fitness):
            return -0.1
        if fitness > 600:
            return -0.1
        return fitness

    def _on_generation_checkpoint(self, session: AsyncSession) -> None:
        session.add(
            DbOptimizerState(
                process_id=self._process_id,
                generation_index=self.generation_index,
                rng=pickle.dumps(self._rng.getstate()),
                innov_db_brain=self._innov_db_brain.Serialize(),
                simulation_time=self._simulation_time,
                sampling_frequency=self._sampling_frequency,
                control_frequency=self._control_frequency,
                num_generations=self._num_generations,
            )
        )


DbBase = declarative_base()


class DbOptimizerState(DbBase):
    __tablename__ = "optimizer"

    process_id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        primary_key=True,
    )
    generation_index = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False, primary_key=True
    )
    rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)
    innov_db_brain = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    simulation_time = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    sampling_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    control_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    num_generations = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
