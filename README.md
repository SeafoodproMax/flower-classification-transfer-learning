# flower-classification-transfer-learning

How does fine-tuning depth interact with training duration
in transfer learning for small datasets?

## Conclusion

On the TF Flowers dataset, fine-tuning only the final ResNet50 stage (stage5) achieved the highest validation accuracy.  
Keeping the backbone fully frozen led to underfitting, while unfreezing deeper layers did not improve performance and sometimes slightly reduced it.

These results suggest that, for small datasets, shallow fine-tuning of the final convolutional block provides the best balance between feature adaptation and overfitting.