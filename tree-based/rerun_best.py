from genotype import GenotypeSerializer, develop
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select

from revolve2.core.database import open_async_database_sqlite
from revolve2.core.database.serializers import DbFloat
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerIndividual
from revolve2.runners.isaacgym import ModularRobotRerunner


async def main() -> None:
    top_n = 30
    db = open_async_database_sqlite("/home/ibilgin/Desktop/ilayda-tree/tree-database")
    async with AsyncSession(db) as session:
        top_n_individuals = (
            await session.execute(
                select(DbEAOptimizerIndividual, DbFloat)
                .filter(DbEAOptimizerIndividual.fitness_id == DbFloat.id)
                .order_by(DbFloat.value.desc()).limit(top_n)
            )
        )

        assert top_n_individuals is not None

        genotypes = []
        for individual in top_n_individuals:
            print(f"fitness: {individual[1].value}")
            genotype = (
                await GenotypeSerializer.from_database(
                    session, [individual[0].genotype_id]
                )
            )[0]
            # print(genotype)
            developed, morphology, max_parts = develop(genotype)
            genotypes.append(developed)


        # genotype = (
        #     await GenotypeSerializer.from_database(
        #         session, [best_individual[0].genotype_id]
        #     )
        # )[0]
    #
    rerunner = ModularRobotRerunner()
    await rerunner.rerun(genotypes, 30)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
