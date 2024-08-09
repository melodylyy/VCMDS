# VCMDS
MiRNA-drug sensitivity prediction by variational graph auto-encoders and collaborative matrix factorization.

The programs is supported by Python 3.6 . 
# Input
* miRNA-drug sensitivity association network
* miRNA-miRNA association network
* drug-drug associations network
# Method
VCMDS calculates Gaussian Interaction Profile (GIP) kernel similarities for miRNAs and drugs, integrating these metrics into their respective similarity networks. We variational graph autoencoders and collaborative matrix factorization CMF to extract features. Finally, predicted scores are obtained using a fully connected network.

