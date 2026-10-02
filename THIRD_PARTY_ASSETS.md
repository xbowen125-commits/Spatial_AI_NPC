# Third-party assets

This repository contains integration code only. It does not include the Airi VRM avatar, Mixamo animation files, UniVRM package binaries, model weights, or downloaded speech-recognition models.

## VRoid / VRM avatars

Create or obtain a VRM 1.0 avatar from a source whose license permits your intended use. VRoid characters and user-created VRM files may have creator-specific conditions. Do not redistribute an avatar merely because it can be imported into Unity.

The development scene used a locally owned VRoid VRM 1.0 avatar named `Airi_v0`. That file is intentionally excluded from this repository.

## UniVRM

The tested Unity project uses UniVRM `v0.131.2`. Install it separately in the Unity project through the official UniVRM package or its documented Unity Package Manager Git URLs. UniVRM remains subject to its own license and notices.

## Mixamo animations

The development scene used a locally downloaded Mixamo `Breathing Idle` animation. The FBX is intentionally excluded. Download the animation through your own Adobe/Mixamo account and follow the applicable Mixamo terms.

## Local AI models

`faster-whisper` downloads or loads speech-recognition model files separately. Model weights and caches are not part of this repository. Review the model's license before redistribution or deployment.

Before publishing a fork, verify every avatar, animation, model, texture, font, audio file, and Unity package independently. Do not assume that a public GitHub repository grants redistribution rights for third-party assets.
