# To make new registries

Registry file name convention:
[manufacturer]_[test_method].toml

- Use lower case.  
- Underscore (`_`) between manufacturer and test_method.
- Hyphen (`-`) between words.

File Name Examples:

| Raw File / Instrument | Target TOML Registry Name |
| --- | --- |
| Anton Paar MCR Rheometer | `anton-paar_rheology.toml` |
| TA Instruments Discovery DMA | `ta-instruments_dma.toml` |
| PerkinElmer Pyris TGA | `perkinelmer_tga.toml` |