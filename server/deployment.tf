provider "google" {
  region = "us-central1"
}

variable "GOOGLE_PROJECT" {}

# cluster

resource "google_compute_instance_template" "pix2pix" {
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
    ExecStart=/usr/bin/docker run --log-driver=gcplogs --restart always -u 2000 --publish 80:8080 --name=pix2pix us.gcr.io/${var.GOOGLE_PROJECT}/pix2pix-server:v3 python -u serve.py --port 8080 --local_models_dir models --cloud_model_names facades_BtoA,edges2cats_AtoB,edges2shoes_AtoB,edges2handbags_AtoB
    ExecStop=/usr/bin/docker stop pix2pix
    ExecStopPost=/usr/bin/docker rm pix2pix

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

resource "google_compute_http_health_check" "pix2pix" {
  name         = "pix2pix-check"
  request_path = "/health"

  timeout_sec        = 5
  check_interval_sec = 5
}

resource "google_compute_target_pool" "pix2pix" {
  name = "pix2pix-pool"

  health_checks = [
    "${google_compute_http_health_check.pix2pix.name}",
  ]
}

resource "google_compute_instance_group_manager" "pix2pix" {
  name               = "pix2pix-manager"
  instance_template  = "${google_compute_instance_template.pix2pix.self_link}"
  base_instance_name = "pix2pix"
  zone               = "us-central1-c"

  target_pools = ["${google_compute_target_pool.pix2pix.self_link}"]

  // don't update instances with terraform, which supposedly can't do a rolling restart
  // use this to update them instead:
  // gcloud compute instance-groups managed recreate-instances pix2pix-manager --zone us-central1-c --instances pix2pix-frfh
  update_strategy = "NONE"
}

resource "google_compute_address" "pix2pix" {
  name = "pix2pix-address"
}

resource "google_compute_forwarding_rule" "pix2pix" {
  name       = "pix2pix-balancer"
  target     = "${google_compute_target_pool.pix2pix.self_link}"
  port_range = "80-80"
  ip_address = "${google_compute_address.pix2pix.address}"
}

resource "google_compute_autoscaler" "pix2pix" {
  name   = "pix2pix-autoscaler"
  zone   = "us-central1-c"
  target = "${google_compute_instance_group_manager.pix2pix.self_link}"

  autoscaling_policy = {
    max_replicas    = 0
    min_replicas    = 0
    cooldown_period = 60

    cpu_utilization {
      target = 0.9
    }
  }
}

# singleton

resource "google_compute_instance" "pix2pix-singleton" {
  name         = "pix2pix-singleton"
  machine_type = "g1-small"
  zone         = "us-central1-c"

  tags           = ["http-server"]
  can_ip_forward = false

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
  }

  disk {
    image = "cos-cloud/cos-stable"
  }

  network_interface {
    network = "default"

    access_config {
      nat_ip = "${google_compute_address.pix2pix-singleton.address}"
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
    ExecStart=/usr/bin/docker run --log-driver=gcplogs --restart always -u 2000 --publish 80:8080 --name=pix2pix us.gcr.io/${var.GOOGLE_PROJECT}/pix2pix-server python -u serve.py --port 8080 --cloud_model_names facades_BtoA,edges2cats_AtoB,edges2shoes_AtoB,edges2handbags_AtoB
    ExecStop=/usr/bin/docker stop pix2pix
    ExecStopPost=/usr/bin/docker rm pix2pix

runcmd:
- iptables -A INPUT -p tcp --dport 80 -j ACCEPT
- systemctl daemon-reload
- systemctl start pix2pix.service
EOF
  }

  service_account {
    scopes = ["https://www.googleapis.com/auth/logging.write", "https://www.googleapis.com/auth/devstorage.read_only", "https://www.googleapis.com/auth/cloud-platform"]
  }
}

resource "google_compute_address" "pix2pix-singleton" {
  name = "pix2pix-singleton"
}
