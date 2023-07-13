#!/bin/sh

sh launch_tinyimagenet.sh er 4000
sh launch_linear_probing.sh /raid/carta/ocl_survey/results/

sh launch_tinyimagenet.sh er_ace 4000
sh launch_linear_probing.sh /raid/carta/ocl_survey/results/er_ace_split_tinyimagenet_20_4000

sh launch_cifar100.sh mir 4000
sh launch_linear_probing.sh /raid/carta/ocl_survey/results/mir_split_tinyimagenet_20_4000

sh launch_cifar100.sh er_lwf 4000
sh launch_linear_probing.sh /raid/carta/ocl_survey/results/er_lwf_split_tinyimagenet_20_4000

sh launch_cifar100.sh rar 4000
sh launch_linear_probing.sh /raid/carta/ocl_survey/results/rar_split_tinyimagenet_20_4000

sh launch_cifar100.sh der 4000
sh launch_linear_probing.sh /raid/carta/ocl_survey/results/der_split_tinyimagenet_20_4000

sh launch_cifar100.sh agem 4000
sh launch_linear_probing.sh /raid/carta/ocl_survey/results/agem_split_tinyimagenet_20_4000

