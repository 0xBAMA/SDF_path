# SDFs
Experimenting with SDFs - Built on NQAE

## Information on the Previous Iterations of this Project:
- [Part 1: Basic bitcrush dithering in the RGB colorspace](https://jbaker.graphics/writings/sdf1.html)
- [Part 2: More experiments with SDF techniques and dithering in different colorspaces](https://jbaker.graphics/writings/sdf2.html)

### Plans for part 3:
This has been a little while coming. I've been thinking a lot about what I want to do with it, while dealing with the ongoing feature creep of Voraldo 1.2. On the plus side, that's been a good opportunity to get up to speed on SDF methodology (folds and other space manipulation, plus a huge range of starting primitives).

The raymarching code is in resources > engine_code > shaders > raymarch.cs.glsl

The dithering code is in resoureces > engine_code > shaders > dither.cs.glsl

- ~~dithering in even more color spaces,~~ perhaps blending a few different results together?
- finish implementing more flexible bitcrush implementation (considering signed and unsigned 8-bit values to deal with colorspaces defined in negative ranges)
- ~~soft shadows~~
- ~~maybe higher resolution (640x480 or widescreen variant of that?) - currently 512x256, just to be friendly with shader invocation dimensions~~
- ~~multisampling the initial raymarching result~~
- ~~experimenting with distance estimated fractals~~
- ~~a scheme for refractive objects - lens shapes made by the intersection of two spheres (spheres of different radii?) - methodology based on hit, refract, march to hit on the inside (using inverted lens SDF), refract again on exit, then march to final hit - this is all a relatively hard-coded process the way I have it planned, so we'll see how it turns out~~
- ~~cycled blue noise dithering using the golden ratio trick~~
