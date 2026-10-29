# MinecraftMapper
Minecraft world map generator.

## Finding blocks

`--find` lists the location of specific blocks (ores, spawners, chests and so on) in every
chunk that has finished ore/feature generation. Results are grouped into veins (touching
blocks of the same type, with stone and deepslate variants counted together) and written
to a CSV. Found veins are also marked on the map unless `--find-only` is used.

```
# Diamonds within 500 blocks of your base, nearest first, CSV only
python mcmap.py <world or region folder> --find diamond_ore,deepslate_diamond_ore --near 120 -340 --radius 500 --find-only

# Same, using a wildcard, and mark them on the map
python mcmap.py <world> --find "*diamond_ore" --near 120 -340 --radius 500

# Ancient debris in the Nether, one CSV row per block
python mcmap.py <world> -d nether --find ancient_debris --near 0 0 --every-block
```

| Option | Meaning |
| --- | --- |
| `--find BLOCKS` | Comma-separated block names; `*` and `?` wildcards allowed |
| `--find-out FILE` | CSV path (default `<world>_<dimension>_found.csv`) |
| `--every-block` | One CSV row per block instead of per vein |
| `--near X Z` | Sort by horizontal distance from this point |
| `--radius R` | With `--near`, keep only results within R blocks |
| `--min-y` / `--max-y` | Limit the height range searched |
| `--find-only` | Skip rendering the map (with `--radius`, only nearby chunks are read) |
| `--no-find-markers` / `--find-color` | Turn off or recolor the map markers |

Each vein row's `x, y, z` is a real ore block near the middle of the vein, so you can go
straight to it.
