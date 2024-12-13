#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// Función para detectar globos por color y dibujar contornos
int detectarGlobos(const Mat& imagenOriginal, Mat& imagenResultado, const string& colorSeleccionado) {
    Mat imagenHSV;
    cvtColor(imagenOriginal, imagenHSV, COLOR_BGR2HSV);

    // Rango de color en HSV para cada color
    Mat mascara;
    if (colorSeleccionado == "rojo") {
        Mat mask1, mask2;
        inRange(imagenHSV, Scalar(0, 100, 100), Scalar(10, 255, 255), mask1);
        inRange(imagenHSV, Scalar(170, 100, 100), Scalar(180, 255, 255), mask2);
        add(mask1, mask2, mascara);
    }
    else if (colorSeleccionado == "verde") {
        inRange(imagenHSV, Scalar(35, 100, 100), Scalar(85, 255, 255), mascara);
    }
    else if (colorSeleccionado == "azul") {
        inRange(imagenHSV, Scalar(90, 50, 50), Scalar(130, 255, 255), mascara);
    }
    else {
        cerr << "Color no válido. Por favor, selecciona rojo, verde o azul." << endl;
        return -1;
    }

    // Realizar operaciones morfológicas para limpiar el ruido
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mascara, mascara, MORPH_CLOSE, kernel);

    // Encontrar contornos
    vector<vector<Point>> contornos;
    findContours(mascara, contornos, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Dibujar contornos en la imagen de resultado
    imagenResultado = imagenOriginal.clone();
    int contadorGlobos = 0;
    for (const auto& contorno : contornos) {
        if (contourArea(contorno) > 500) { // Filtrar contornos pequeños
            drawContours(imagenResultado, vector<vector<Point>>{contorno}, -1, Scalar(0, 255, 128), 3); // Verde claro
            contadorGlobos++;
        }
    }

    return contadorGlobos;
}

// Función para convertir la imagen a escala de grises
Mat grayScale(Mat img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    return gray;
}

// Función para realizar sampling
Mat sampling(Mat img, int factor) {
    Mat sampled;
    resize(img, sampled, Size(), factor / 100.0, factor / 100.0, INTER_LINEAR);
    resize(sampled, sampled, img.size(), 0, 0, INTER_NEAREST);
    return sampled;
}

// Función para realizar cuantización
Mat quantize(Mat img, int level) {
    Mat quantized;
    img.convertTo(quantized, CV_8U, 1.0 / level, 0);
    quantized.convertTo(quantized, CV_8U, level, 0);
    return quantized;
}

int main() {
    // Leer imagen
    Mat img = imread("Img/FIGA.jpg");
    if (img.empty()) {
        cout << "Error al cargar la imagen FIGA.jpg" << endl;
        return -1;
    }

    // Preguntar al usuario
    string colorSeleccionado;
    int samplingValue, quantizationValue;
    cout << "Escriba el color a filtrar (rojo, verde, azul): ";
    cin >> colorSeleccionado;
    cout << "Escriba el valor de sampling: ";
    cin >> samplingValue;
    cout << "Escriba el valor de cuantizacion: ";
    cin >> quantizationValue;

    // Detección de globos
    Mat contouredImage;
    int balloonCount = detectarGlobos(img, contouredImage, colorSeleccionado);

    // Mostrar imágenes
    imshow("Imagen Original", img);
    if (!contouredImage.empty())
        imshow("Globos Detectados", contouredImage);

    Mat grayImg = grayScale(img);
    imshow("Imagen en Escala de Grises", grayImg);

    Mat sampledImg = sampling(img, samplingValue);
    imshow("Imagen con Sampling", sampledImg);

    Mat quantizedImg = quantize(img, quantizationValue);
    imshow("Imagen con Cuantizada", quantizedImg);

    // Resumen en consola
    cout << "Resumen:\n";
    cout << "Cantidad de globos encontrados: " << balloonCount << "\n";
    cout << "Tamaño de la imagen: " << img.cols << " x " << img.rows << " pixeles\n";
    cout << "Valor de sampling seleccionado: " << samplingValue << "\n";
    cout << "Valor de cuantizacion seleccionado: " << quantizationValue << "\n";

    waitKey(0);
    return 0;
}
