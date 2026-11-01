# FastWAM on LIBERO

MAD models for the hands-on post about skills, agents, and workflows.
The image is `docker/pyt_fam.ubuntu.amd.Dockerfile`. `run.sh` calls the
FastWAM Python entry points inside that image and does not start Docker.

```bash
madengine run --tags pyt_fam_fastwam_latency
```

`MAD_SECRETS_HFTOKEN` is required. Checkpoints download on first run.
