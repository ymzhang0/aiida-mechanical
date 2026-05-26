def plot_group(
    ax,
    group: orm.Group|str,
    keys: list[str],
    **kwargs,
):
    if isinstance(group, str):
        group = orm.Group.collection.get(group)

    for node in group.nodes:
        for key in keys:
            if key in node.extras:
                ax.plot(node.extras[key], **kwargs)
    return ax
