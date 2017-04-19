# pix2pix-tensorflow server

Host pix2pix-tensorflow models to be used with something like the [Image-to-Image Demo](https://affinelayer.com/pixsrv/).

This is a simple python server that serves models exported from `pix2pix.py --mode export`.  It can serve local models or use [Cloud ML](https://cloud.google.com/ml/) to run the model.

## Exporting

You can export a model to be served with `--mode export`. As with testing, you should specify the checkpoint to use with `--checkpoint`.

```sh
python ../pix2pix.py \
  --mode export \
  --output_dir models/facades \
  --checkpoint ../facades_train
```

## Local Serving

Using the [pix2pix-tensorflow Docker image](https://hub.docker.com/r/affinelayer/pix2pix-tensorflow/):

```sh
# export a model to upload (if you did not export one above)
python ../tools/dockrun.py python tools/export-example-model.py --output_dir models/example
# process an image with the model using local tensorflow
python ../tools/dockrun.py python tools/process-local.py \
    --model_dir models/example \
    --input_file static/facades-input.png \
    --output_file output.png
# run local server
python ../tools/dockrun.py --port 8000 python serve.py --port 8000 --local_models_dir models
# test the local server
python tools/process-remote.py \
    --input_file static/facades-input.png \
    --url http://localhost:8000/example \
    --output_file output.png
```

If you open [http://localhost:8000/](http://localhost:8000/) in a browser, you should see an interactive demo, though this expects the server to be hosting the exported models available here:

- [edges2shoes](https://mega.nz/#!HtYwAZTY!5tBLYt_6HFj9u2Kxgp4-I36O4EV9r3bDP44ztX3qesI)
- [edges2handbags](https://mega.nz/#!Clg3EaLA!YW2jfRHvwpJn5Elww_wM-f3eRzKiGHLw-F4A3eQCceI)
- [facades](https://mega.nz/#!f1ZjmZoa!mCSxFRxt1WLBpNFsv5raoroEigxomDVpdi40aOG1KMc)

Extract those to the models directory and restart the server to have it host the models.

## Cloud ML Serving

For this you'll want to generate a service account JSON file from https://console.cloud.google.com/iam-admin/serviceaccounts/project (select "Furnish a new private key").  If you are already logged in with the gcloud SDK, the script will auto-detect credentials from that if you leave off the `--credentials` option.

```sh
# upload model to google cloud ml
python ../tools/dockrun.py python tools/upload-model.py \
    --bucket your-models-bucket-name-here \
    --model_name example \
    --model_dir models/example \
    --credentials service-account.json
# process an image with the model using google cloud ml
python ../tools/dockrun.py python tools/process-cloud.py \
    --model example \
    --input_file static/facades-input.png \
    --output_file output.png \
    --credentials service-account.json
```

## Running serve.py on Google Cloud Platform

Assuming you have gcloud and docker setup:

```sh
export GOOGLE_PROJECT=<project name>
# build image
# make sure models are in a directory called "models" in the current directory
sudo docker build --rm --tag us.gcr.io/$GOOGLE_PROJECT/pix2pix-server .
# test image locally
sudo docker run --publish 8080:8080 --rm --name server us.gcr.io/$GOOGLE_PROJECT/pix2pix-server python -u serve.py \
    --port 8080 \
    --local_models_dir models
python tools/process-remote.py \
    --input_file static/facades-input.png \
    --url http://localhost:8080/example \
    --output_file output.png
# publish image to private google container repository
python tools/upload-image.py --project $GOOGLE_PROJECT --version v1
# setup server
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars to put your cloud info in there
python ../tools/dockrun.py terraform plan
python ../tools/dockrun.py terraform apply
```

## Full training + exporting + hosting commands

Tested with Python 3.6, Tensorflow 1.0.0, Docker, gcloud, and Terraform (https://www.terraform.io/downloads.html)

```sh
git clone https://github.com/affinelayer/pix2pix-tensorflow.git
cd pix2pix-tensorflow

# get some images (only 2 for testing)
mkdir source
curl -o source/cat1.jpg https://farm5.staticflickr.com/4032/4394955222_eea73818d9_o.jpg
curl -o source/cat2.jpg http://wallpapercave.com/wp/ePMeSmp.jpg

# resize source images
python tools/process.py \
  --input_dir source \
  --operation resize \
  --output_dir resized

# create edges from resized images (uses docker container since compiling the dependencies is annoying)
python tools/dockrun.py python tools/process.py \
  --input_dir resized \
  --operation edges \
  --output_dir edges

# combine resized with edges
python tools/process.py \
  --input_dir edges \
  --b_dir resized \
  --operation combine \
  --output_dir combined

# train on images (only 1 epoch for testing)
python pix2pix.py \
  --mode train \
  --output_dir train \
  --max_epochs 1 \
  --input_dir combined \
  --which_direction AtoB

# export model (creates a version of the model that works with the server in server/serve.py as well as google hosted tensorflow)
python pix2pix.py \
  --mode export \
  --output_dir server/models/edges2cats_AtoB \
  --checkpoint train

# process image locally using exported model
python server/tools/process-local.py \
    --model_dir server/models/edges2cats_AtoB \
    --input_file edges/cat1.png \
    --output_file output.png

# serve model locally
cd server
python serve.py --port 8000 --local_models_dir models

# open http://localhost:8000 in a browser, and scroll to the bottom, you should be able to process an edges2cat image and get a bunch of noise as output

# serve model remotely

export GOOGLE_PROJECT=<project name>

# build image
# make sure models are in a directory called "models" in the current directory
docker build --rm --tag us.gcr.io/$GOOGLE_PROJECT/pix2pix-server .

# test image locally
docker run --publish 8000:8000 --rm --name server us.gcr.io/$GOOGLE_PROJECT/pix2pix-server python -u serve.py \
    --port 8000 \
    --local_models_dir models

# run this while the above server is running
python tools/process-remote.py \
    --input_file static/edges2cats-input.png \
    --url http://localhost:8000/edges2cats_AtoB \
    --output_file output.png

# publish image to private google container repository
python tools/upload-image.py --project $GOOGLE_PROJECT --version v1

# create a google cloud server
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars to put your cloud info in there
# get the service-account.json from the google cloud console
# make sure GCE is enabled on your account as well
python terraform plan
python terraform apply

# get name of server
gcloud compute instance-groups list-instances pix2pix-manager
# ssh to server
gcloud compute ssh <name of instance here>
# look at the logs (can take awhile to load docker image)
sudo journalctl -f -u pix2pix
# if you have never made an http-server before, apparently you may need this rule
gcloud compute firewall-rules create http-server --allow=tcp:80 --target-tags http-server
# get ip address of load balancer
gcloud compute forwarding-rules list
# open that in the browser, should see the same page you saw locally

# to destroy the GCP resources, use this
terraform destroy
```