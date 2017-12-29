# pix2pix-tensorflow server

Host pix2pix-tensorflow models to be used with something like the [Image-to-Image Demo](https://affinelayer.com/pixsrv/).

This is a simple python server that uses [deeplearn.js](https://deeplearnjs.org/) and weights exported from pix2pix checkpoints using `tools/export-checkpoint.py`.

## Exporting

You can export a model to be served with `tools/export-checkpoint.py`.

```sh
python tools/export-checkpoint.py \
  --checkpoint facades_BtoA \
  --output_file static/models/facades_BtoA.bin
```

You can also copy models from the `pix2pix-tensorflow-models` repo:

```sh
git clone git@github.com:affinelayer/pix2pix-tensorflow-models.git static/models
```

## Serving

```sh
python serve.py --port 8000
```

If you open [http://localhost:8000/](http://localhost:8000/) in a browser, you should see an interactive demo.
