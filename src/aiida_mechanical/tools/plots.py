from aiida import orm


def plot_moduli_group(
    ax,
    group: str,
    keys: list[str],
    formula: str = "vrh",
    **kwargs,
):
    if isinstance(group, str):
        qb = orm.QueryBuilder()
        qb.append(
            orm.Group,
            filters={"label": group},
            tag="group",
        ).append(
            orm.WorkChainNode,
            with_group="group",
            filters={"attributes.exit_status": 0},
            tag="wc",
        )
    else:
        raise ValueError("Group name must be str")

    print(f"Found {qb.count()} workchains")
    for key in keys:
        x = []
        y = []
        for node in qb.all(flat=True):
            x.append(
                node.outputs.output_parameters.get("moduli").get(formula).get(keys[0])
            )
            y.append(
                node.outputs.output_parameters.get("moduli").get(formula).get(keys[1])
            )
        ax.scatter(x, y, **kwargs)
