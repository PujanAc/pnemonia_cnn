"""
Database models for storing predictions and uploaded images.
"""

from django.db import models
from django.utils import timezone


class Prediction(models.Model):
    """
    Store prediction results for tracking and display.
    """
    MODEL_CHOICES = [
        ('custom_cnn', 'Custom CNN'),
        ('resnet50', 'ResNet-50'),
    ]
    
    RESULT_CHOICES = [
        ('normal', 'Normal'),
        ('pneumonia', 'Pneumonia'),
    ]
    
 
    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    model_used = models.CharField(max_length=20, choices=MODEL_CHOICES)
    

    predicted_class = models.CharField(max_length=20, choices=RESULT_CHOICES)
    confidence = models.FloatField()  

    gradcam_image = models.ImageField(upload_to='gradcam/%Y/%m/%d/', null=True, blank=True)

    created_at = models.DateTimeField(default=timezone.now)
    processing_time = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Prediction'
        verbose_name_plural = 'Predictions'
    
    def __str__(self):
        return f"{self.model_used} - {self.predicted_class} ({self.confidence:.2f}%)"
    
    def get_confidence_color(self):
        """Return color based on confidence level."""
        if self.confidence >= 90:
            return 'success'
        elif self.confidence >= 70:
            return 'warning'
        else:
            return 'danger'


class ComparisonResult(models.Model):
    """
    Store comparison results when both models are run.
    """
    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    
    # Custom CNN results
    cnn_prediction = models.CharField(null=True,blank=True,max_length=20)
    cnn_confidence = models.FloatField(null=True,blank=True)
    cnn_gradcam = models.ImageField(upload_to='gradcam/%Y/%m/%d/', null=True, blank=True)
    
    # ResNet-50 results
    resnet_prediction = models.CharField(max_length=20, null=True, blank=True)
    resnet_confidence = models.FloatField( null=True, blank=True)
    resnet_gradcam = models.ImageField(upload_to='gradcam/%Y/%m/%d/', null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    total_processing_time = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Comparison Result'
        verbose_name_plural = 'Comparison Results'
    
    def __str__(self):
        return f"Comparison - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    def models_agree(self):
        """Check if both models predicted the same class."""
        return self.cnn_prediction == self.resnet_prediction