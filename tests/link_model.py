import typer
import os
import wandb


def link_model(artifact_path: str, aliases: list[str] = ["staging"]) -> None:

    if artifact_path == "":
        typer.echo("No artifact path provided. Exiting.")
        return

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    _, _, artifact_name_version = artifact_path.split("/")
    artifact_name, _ = artifact_name_version.split(":")

    artifact = api.artifact(artifact_path)
    artifact.link(target_path=f"{os.getenv('WANDB_ENTITY')}/model-registry/{artifact_name}", aliases=aliases)
    artifact.save()
    typer.echo(f"Artifact {artifact_path} linked to {aliases}")

if __name__ == "__main__":
    typer.run(link_model)