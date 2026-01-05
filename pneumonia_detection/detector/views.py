import time
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.urls import reverse
from django.conf import settings

from .forms import ModelSelectionForm, ImageUploadForm, CompareUploadForm
from .models import Prediction, ComparisonResult
from .utils import ModelPredictor, generate_gradcam, save_gradcam_to_file


def home(request):
    return render(request, 'detector/home.html', {
        'total_predictions': Prediction.objects.count(),
        'model_info': settings.MODEL_INFO,
    })


def select_model(request):
    if request.method == 'POST':
        form = ModelSelectionForm(request.POST)
        if form.is_valid():
            model_type = form.cleaned_data['model']
            return redirect('detector:upload', model_type=model_type)
    else:
        form = ModelSelectionForm()
    return render(request, 'detector/select_model.html', {
        'form': form,
        'model_info': settings.MODEL_INFO,
    })


def upload_image(request, model_type):
    if model_type not in ['custom_cnn', 'resnet50']:
        messages.error(request, "Invalid model type.")
        return redirect('detector:select_model')

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            prediction = Prediction.objects.create(
                image=image,
                model_used=model_type,
                predicted_class='normal',
                confidence=0.0
            )
            url = reverse('detector:predict')
            return redirect(f"{url}?pred_id={prediction.id}")
    else:
        form = ImageUploadForm()
    return render(request, 'detector/upload.html', {
        'form': form,
        'model_type': model_type,
        'model_info': settings.MODEL_INFO[model_type],
    })


def predict(request):
    pred_id = request.GET.get('pred_id')
    if not pred_id:
        messages.error(request, "No image uploaded.")
        return redirect('detector:select_model')

    prediction = get_object_or_404(Prediction, id=pred_id)
    start_time = time.time()

    try:
        predictor = ModelPredictor(model_type=prediction.model_used)
        image_path = prediction.image.path
        result = predictor.predict(image_path)


        prediction.predicted_class = result['class']
        prediction.confidence = result['confidence']

    
        gradcam_img = None
        if not result.get('mock', False):
            gradcam_img = generate_gradcam(predictor, image_path, settings.IMAGE_INPUT_SIZE)

        if gradcam_img:
            gradcam_file = save_gradcam_to_file(gradcam_img)
            if gradcam_file:
                prediction.gradcam_image.save(f"gradcam_{pred_id}.png", gradcam_file, save=False)

        prediction.processing_time = time.time() - start_time
        prediction.save()

        if result.get('mock', False):
            messages.warning(request, "Demo result shown. Deploy trained model for real predictions.")

        return redirect('detector:result', prediction_id=prediction.id)

    except Exception as e:
        messages.error(request, f"Prediction error: {str(e)}")
        return redirect('detector:select_model')


def result(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id)
    model_info = settings.MODEL_INFO.get(prediction.model_used, {})
    return render(request, 'detector/result.html', {
        'prediction': prediction,
        'model_info': model_info,
        'confidence_color': prediction.get_confidence_color(),
    })


def compare_mode(request):
    if request.method == 'POST':
        form = CompareUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            comparison = ComparisonResult.objects.create(image=image)
            url = reverse('detector:compare_predict')
            return redirect(f"{url}?comp_id={comparison.id}")
    else:
        form = CompareUploadForm()
    return render(request, 'detector/compare.html', {
        'form': form,
        'model_info': settings.MODEL_INFO,
    })


def compare_predict(request):
    comp_id = request.GET.get('comp_id')
    if not comp_id:
        messages.error(request, "No comparison initiated.")
        return redirect('detector:compare')

    comparison = get_object_or_404(ComparisonResult, id=comp_id)
    start_time = time.time()
    try:
        image_path = comparison.image.path

        # Custom CNN
        cnn_predictor = ModelPredictor('custom_cnn')
        cnn_result = cnn_predictor.predict(image_path)
        comparison.cnn_prediction = cnn_result['class']
        comparison.cnn_confidence = cnn_result['confidence']
        if not cnn_result.get('mock', False):
            gradcam = generate_gradcam(cnn_predictor, image_path, settings.IMAGE_INPUT_SIZE)
            if gradcam:
                gradcam_file = save_gradcam_to_file(gradcam)
                if gradcam_file:
                    comparison.cnn_gradcam.save(f"cnn_gradcam_{comp_id}.png", gradcam_file, save=False)

        # ResNet-50
        resnet_predictor = ModelPredictor('resnet50')
        resnet_result = resnet_predictor.predict(image_path)
        comparison.resnet_prediction = resnet_result['class']
        comparison.resnet_confidence = resnet_result['confidence']
        if not resnet_result.get('mock', False):
            gradcam = generate_gradcam(resnet_predictor, image_path, settings.IMAGE_INPUT_SIZE)
            if gradcam:
                gradcam_file = save_gradcam_to_file(gradcam)
                if gradcam_file:
                    comparison.resnet_gradcam.save(f"resnet_gradcam_{comp_id}.png", gradcam_file, save=False)

        comparison.total_processing_time = time.time() - start_time
        comparison.save()

        if cnn_result.get('mock', False) or resnet_result.get('mock', False):
            messages.warning(request, "Demo results shown. Deploy trained models for real predictions.")

        return render(request, 'detector/compare.html', {
            'comparison': comparison,
            'model_info': settings.MODEL_INFO,
            'models_agree': comparison.models_agree(),
        })

    except Exception as e:
        messages.error(request, f"Comparison error: {str(e)}")
        return redirect('detector:compare')
