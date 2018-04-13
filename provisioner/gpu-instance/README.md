# Single instance anaconda provisioner
To export a plan of the provisioning of a single ec2 instance with anaconda:

```bash
terraform plan -var-file=config.tfvars -out=plan
```

to provision the instance:

```bash
terraform apply plan
```

