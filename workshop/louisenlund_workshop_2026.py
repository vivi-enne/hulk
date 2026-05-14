import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import workshop

    from pathlib import Path
    import os
    import marimo as mo
    from mujoco import MjData, MjModel, Renderer, mj_step, mj_resetData, mj_forward
    from workshop import MujocoViewer

    return (
        MjData,
        MjModel,
        MujocoViewer,
        Path,
        Renderer,
        mj_forward,
        mj_resetData,
        mo,
        os,
        workshop,
    )


@app.cell
def _(mo, workshop):
    mo.md(rf"""
    Version {workshop.TEST}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Arm Animation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Der K1 Roboter hat in jedem Arm __3__ Gelenke:
        - Schulter heben
        - Schulter rotieren
        - Ellenbogen anwinkeln

    Alle Gelenke sind *Rotationsgelenke*, sie beschreiben eine Rotation auf einer zwei-dimensionalen Ebene.
    """)
    return


@app.cell
def _(MjData, MjModel, MujocoViewer, Path, Renderer, mj_forward, mo, os):

    os.environ["MUJOCO_GL"] = "glfw"

    model = MjModel.from_xml_path(str(Path("./K1/K1.xml").resolve()))
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    data = MjData(model)
    renderer = Renderer(model, width=1280, height=720)
    viewer = mo.ui.anywidget(MujocoViewer())
    interval = 0.1



    index_shoulder_roll = model.joint("Right_Shoulder_Roll").id
    index_shoulder_pitch = model.joint("ARight_Shoulder_Pitch").id
    index_elbow_yaw = model.joint("Right_Elbow_Yaw").id
    index_elbow_pitch = model.joint("Right_Elbow_Pitch").id

    ctrl_min_shoulder_roll = model.jnt_range[index_shoulder_roll, 0]
    ctrl_max_shoulder_roll = model.jnt_range[index_shoulder_roll, 1]
    ctrl_min_shoulder_pitch = model.jnt_range[index_shoulder_pitch, 0]
    ctrl_max_shoulder_pitch = model.jnt_range[index_shoulder_pitch, 1]
    ctrl_min_elbow_yaw = model.jnt_range[index_elbow_yaw, 0]
    ctrl_max_elbow_yaw = model.jnt_range[index_elbow_yaw, 1]
    ctrl_min_elbow_pitch = model.jnt_range[index_elbow_pitch, 0]
    ctrl_max_elbow_pitch = model.jnt_range[index_elbow_pitch, 1]

    # slider for joint movement
    slider_shoulder_roll = mo.ui.slider(start=ctrl_min_shoulder_roll, stop=ctrl_max_shoulder_roll, label="Schulter roll", value=0, step= 0.1)
    slider_shoulder_pitch = mo.ui.slider(start=ctrl_min_shoulder_pitch, stop=ctrl_max_shoulder_pitch, label="Schulter pitch", value=0, step=0.1)
    slider_elbow_yaw = mo.ui.slider(start=ctrl_min_elbow_yaw, stop=ctrl_max_elbow_yaw, label="Ellbogen yaw", value=0, step=0.1)
    slider_elbow_pitch = mo.ui.slider(start=ctrl_min_elbow_pitch, stop=ctrl_max_elbow_pitch, label="Ellbogen pitch", value=0, step=0.1)

    mj_forward(model, data)
    renderer.update_scene(data, camera="overview_cam")
    initial_pixels = renderer.render()
    viewer.update(initial_pixels)
    return (
        data,
        index_elbow_pitch,
        index_elbow_yaw,
        index_shoulder_pitch,
        index_shoulder_roll,
        interval,
        model,
        renderer,
        slider_elbow_pitch,
        slider_elbow_yaw,
        slider_shoulder_pitch,
        slider_shoulder_roll,
        viewer,
    )


@app.cell
def _(
    MjData,
    MjModel,
    data,
    index_elbow_pitch,
    index_elbow_yaw,
    index_shoulder_pitch,
    index_shoulder_roll,
    interval,
    mj_forward,
    mj_resetData,
    mo,
    model,
    renderer,
    slider_elbow_pitch,
    slider_elbow_yaw,
    slider_shoulder_pitch,
    slider_shoulder_roll,
    viewer,
):
    def advance_simulation(mj_model: MjModel,mj_data: MjData, dt: float) -> None:
        mj_forward(mj_model, mj_data)


    def update(_):
        advance_simulation(model, data, interval)
        renderer.update_scene(data, camera="overview_cam")
        rendered_pixels = renderer.render()
        viewer.update(rendered_pixels)

    def reset_simulation(_):
        mj_resetData(model, data)


    data.qpos[index_shoulder_roll] = slider_shoulder_roll.value
    data.qpos[index_shoulder_pitch] = slider_shoulder_pitch.value 
    data.qpos[index_elbow_yaw] = slider_elbow_yaw.value
    data.qpos[index_elbow_pitch] = slider_elbow_pitch.value

    mj_forward(model, data)

    renderer.update_scene(data, camera="overview_cam")
    rendered_pixels = renderer.render()
    viewer.update(rendered_pixels)

    restart_btn = mo.ui.button(label="🔄 Simulation neustarten", on_change=reset_simulation)
    refresh_timer = mo.ui.refresh(default_interval=interval, on_change=update)

    mo.vstack(
    [
       mo.hstack(
           [
               mo.vstack([slider_shoulder_roll, slider_shoulder_pitch, slider_elbow_yaw, slider_elbow_pitch]),
               refresh_timer, restart_btn,
           ]
       ),
       viewer,
    ])
    return


if __name__ == "__main__":
    app.run()
