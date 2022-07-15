"""
Rerun(watch) a modular robot in isaac gym.
"""

from pyrr import Quaternion, Vector3

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor
from revolve2.runners.isaacgym import LocalRunner


class ModularRobotRerunner:
    _controller: ActorController

    async def rerun(self, robot: ModularRobot, control_frequency: float) -> None:
        batch = Batch(
            simulation_time=20,
            sampling_frequency=5,
            control_frequency=control_frequency,
            control=self._control,
        )

        # batch = Batch(
        #     simulation_time=1000000,
        #     sampling_frequency=0.0001,
        #     control_frequency=control_frequency,
        #     control=self._control,
        # )

        self._controllers = []

        for genotype in robot:
            actor, controller = genotype.make_actor_and_controller()


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

        runner = LocalRunner(LocalRunner.SimParams())
        await runner.run_batch(batch)

    def _control(self, dt: float, control: ActorControl) -> None:
        for control_i, controller in enumerate(self._controllers):
            controller.step(dt)
            control.set_dof_targets(control_i, 0, controller.get_dof_targets())


if __name__ == "__main__":
    print(
        "This file cannot be ran as a script. Import it and use the contained classes instead."
    )
