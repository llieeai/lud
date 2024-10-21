
### teacher model ###

python tools/train.py --cfg configs/cifar100/teacher/wrn_40_2.yaml

python tools/train.py --cfg configs/cifar100/teacher/res32x4.yaml

python tools/train.py --cfg configs/cifar100/teacher/res110.yaml

python tools/train.py --cfg configs/cifar100/teacher/vgg13.yaml

python tools/train.py --cfg configs/cifar100/teacher/res50.yaml


### move files ###

mv output/cifar100_teacher/teacher,wrn_40_2/student_best download_ckpts/cifar_teachers/wrn_40_2_vanilla/ckpt_epoch_240.pth

mv output/cifar100_teacher/teacher,res32x4/student_best download_ckpts/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth

mv output/cifar100_teacher/teacher,res110/student_best download_ckpts/cifar_teachers/resnet110_vanilla/ckpt_epoch_240.pth

mv output/cifar100_teacher/teacher,vgg13/student_best download_ckpts/cifar_teachers/vgg13_vanilla/ckpt_epoch_240.pth

mv output/cifar100_teacher/teacher,res50/student_best download_ckpts/cifar_teachers/ResNet50_vanilla/ckpt_epoch_240.pth




