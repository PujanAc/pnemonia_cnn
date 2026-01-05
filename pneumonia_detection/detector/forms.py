"""
Forms for image upload and model selection.
"""

from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError


class ModelSelectionForm(forms.Form):
    """
    Form for selecting which ML model to use.
    """
    MODEL_CHOICES = [
        ('custom_cnn', 'Custom CNN - Lightweight and Fast'),
        ('resnet50', 'ResNet-50 - High Accuracy'),
    ]
    
    model = forms.ChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.RadioSelect(attrs={
            'class': 'model-selector'
        }),
        label='Select AI Model',
        initial='custom_cnn'
    )


class ImageUploadForm(forms.Form):
    """
    Form for uploading chest X-ray images.
    """
    image = forms.ImageField(
        label='Upload Chest X-ray Image',
        help_text='Accepted formats: JPG, PNG (Max size: 10MB)',
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/jpeg,image/png,image/jpg'
        })
    )
    
    def clean_image(self):
        """
        Validate uploaded image.
        """
        image = self.cleaned_data.get('image')
        
        if not image:
            raise ValidationError('Please upload an image.')
        
        # Check file size
        if image.size > settings.MAX_UPLOAD_SIZE:
            raise ValidationError(
                f'Image size must be less than {settings.MAX_UPLOAD_SIZE / (1024*1024):.0f}MB'
            )
        
        # Check file type
        if image.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise ValidationError(
                'Invalid image format. Please upload JPG or PNG files only.'
            )
        
        # Check image dimensions (optional)
        try:
            from PIL import Image
            img = Image.open(image)
            width, height = img.size
            
            # Ensure minimum dimensions
            if width < 100 or height < 100:
                raise ValidationError(
                    'Image dimensions too small. Minimum size: 100x100 pixels.'
                )
            
            # Reset file pointer after reading
            image.seek(0)
            
        except Exception as e:
            raise ValidationError(f'Invalid image file: {str(e)}')
        
        return image


class CompareUploadForm(forms.Form):
    """
    Form for uploading image for comparison mode.
    """
    image = forms.ImageField(
        label='Upload Chest X-ray for Comparison',
        help_text='This image will be analyzed by both models',
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/jpeg,image/png,image/jpg'
        })
    )
    
    def clean_image(self):
        """
        Validate uploaded image (same validation as ImageUploadForm).
        """
        image = self.cleaned_data.get('image')
        
        if not image:
            raise ValidationError('Please upload an image.')
        
        if image.size > settings.MAX_UPLOAD_SIZE:
            raise ValidationError(
                f'Image size must be less than {settings.MAX_UPLOAD_SIZE / (1024*1024):.0f}MB'
            )
        
        if image.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise ValidationError(
                'Invalid image format. Please upload JPG or PNG files only.'
            )
        
        return image