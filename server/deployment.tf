variable "google_project" {}
variable "google_credentials_file" {}
variable "server_image_version" {}

provider "google" {
  region      = "us-central1"
  credentials = "${file(var.google_credentials_file)}"
  project     = "${var.google_project}"
}

# cluster

resource "google_compute_instance_template" "cluster" {
  name_prefix  = "pix2pix-template-"
  machine_type = "n1-highcpu-2"

  tags           = ["http-server"]
  can_ip_forward = false

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
  }

  disk {
    source_image = "cos-cloud/cos-stable"
  }

  network_interface {
    network = "default"

    access_config {
      // Ephemeral IP
    }
  }

  metadata {
    user-data = <<EOF
#cloud-config

users:
- name: pix2pix
  uid: 2000

write_files:
- path: /etc/systemd/system/pix2pix.service
  permissions: 0644
  owner: root
  content: |
    [Unit]
    Description=Run pix2pix

    [Service]
    Environment="HOME=/home/pix2pix"
    ExecStartPre=/usr/share/google/dockercfg_update.sh
    ExecStart=/usr/bin/docker run --rm --log-driver=gcplogs -u 2000 --publish 80:8080 --name=pix2pix us.gcr.io/${var.google_project}/pix2pix-server:${var.server_image_version} python -u serve.py --port 8080 --local_models_dir models --origin https://affinelayer.com
    ExecStop=/usr/bin/docker stop pix2pix
    ExecStopPost=/usr/bin/docker rm pix2pix
    Restart=always
    RestartSec=30

runcmd:
- iptables -A INPUT -p tcp --dport 80 -j ACCEPT
- systemctl daemon-reload
- systemctl start pix2pix.service
EOF
  }

  service_account {
    scopes = ["https://www.googleapis.com/auth/logging.write", "https://www.googleapis.com/auth/devstorage.read_only", "https://www.googleapis.com/auth/cloud-platform"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "google_compute_http_health_check" "cluster" {
  name         = "pix2pix-check"
  request_path = "/health"

  timeout_sec         = 5
  check_interval_sec  = 10
  unhealthy_threshold = 3
}

resource "google_compute_target_pool" "cluster" {
  name = "pix2pix-pool"

  health_checks = [
    "${google_compute_http_health_check.cluster.name}",
  ]
}

resource "google_compute_instance_group_manager" "cluster" {
  name               = "pix2pix-manager"
  instance_template  = "${google_compute_instance_template.cluster.self_link}"
  base_instance_name = "pix2pix"
  zone               = "us-central1-c"

  target_pools = ["${google_compute_target_pool.cluster.self_link}"]

  // don't update instances with terraform, which supposedly can't do a rolling restart
  // use this to update them instead:
  // gcloud compute instance-groups managed recreate-instances pix2pix-manager --zone us-central1-c --instances pix2pix-frfh
  update_strategy = "NONE"
}

resource "google_compute_address" "cluster" {
  name = "pix2pix-cluster"
}

resource "google_compute_forwarding_rule" "cluster" {
  name       = "pix2pix-balancer"
  target     = "${google_compute_target_pool.cluster.self_link}"
  port_range = "80-80"
  ip_address = "${google_compute_address.cluster.address}"
}

resource "google_compute_autoscaler" "cluster" {
  name   = "pix2pix-autoscaler"
  zone   = "us-central1-c"
  target = "${google_compute_instance_group_manager.cluster.self_link}"

  autoscaling_policy = {
    max_replicas    = 8
    min_replicas    = 1
    cooldown_period = 60

    cpu_utilization {
      target = 0.7
    }
  }
}
