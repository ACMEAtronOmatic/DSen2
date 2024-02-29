## Testing

Find a demo in `demoDSen2.py`.


Use the file `s2_tiles_supres.py` to super-resolve Sentinel-2 tiles, by giving the `.zip` file or the `.xml` file of the unzipped file. You must also provide an output filename, the default is geotiff (extension `.tif`).

For Doug: to test the serialized model:
`python3 load_serialized_and_test.py ../training/for_doug_Keras_PB/ ../data/paired/validation/`
