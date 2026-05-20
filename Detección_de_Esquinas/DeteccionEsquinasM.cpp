#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <vector>
#include <cstdio>
#include <Mosaic.h>

using namespace std;
using namespace cv;


#define STD_MAX 10.   //!< El valor de desviación estandar máximo.
#define S_SLIDE_MAX 512  //!< El número máximo de pasos que tiene el control de barra deslizante asociuada a la desviación estandar usada al derivar la imagene en el GUI.

#define K_MAX 0.15   //!< El valor máximo del parametro del detector de esquinas de Harris.
#define K_SLIDE_MAX 512  //!< El número máximo de pasos que tiene el control de barra deslizante asociado a la constante K del detector de esquinas de Harris en el GUI.

#define W_MAX 50.   //!< El valor máximo del tamaño de la ventana usado por el detector de esquinas de Harris.
#define W_SLIDE_MAX 50  //!< El número máximo de pasos que tiene el control de barra deslizante asociado a la constante K del detector de esquinas de Harris en el GUI.

#define IM_WIDTH 320  //!< El ancho de la imagen.
#define IM_HEIGHT 240 //!< Lo alto de la imagen.

/*!
\struct filterData
\brief Esta estructura agrupa los filtros que se van a aplicar a la
 imagen, la desviación estandar con la que se definen los filtros, y el factor
 de conversion entre el numero de posiciones de la barra deslizante de control
 y el valor que toma la desviacion estandar en un momento dado. La estructura es
 útil pues contiene los datos que son necesarios para modificar el estado del
 programa antes las acciones del usuario usando la barra deslizante.
*/
struct filterData
{
    Mat DGh;  //!< Almacena filtro para estimar la derivada unidimensional de la Gaussiana en direccion horizontal.

    Mat DGv;  //!< Almacena filtro para estimar la derivada unidimensional de la Gaussiana en direccion vertical.

    double s; //!< Almacena la desviación estandar.
    
    double sInc; //!< Almacena la razon de incremento utilizada por el control deslizante

    /*!
    \fn filterData (double stddev = 1.)
    \brief Constructo de la clase, inicializa los atributos 's' y 'sInc'.
    \param stddev el valor con el cual inicializamos la desviación estandar (valor por defect = 1.0)
    */
    filterData (double stddev = 1.)
    {
        s = stddev;
        sInc = STD_MAX/S_SLIDE_MAX;
    }
};

struct HarrisData
{
    int W; //!< El ancho de la ventana de computo.
    double k; //!< Almacena el factor k del detector de esquinas de Harris
    double kInc;//!< Almacena la razon de incremento utilizada por el control deslizante
    HarrisData(int w=3, double val = 0.004)
    {
        W = w;
        k = val;
        kInc = K_MAX / K_SLIDE_MAX;
    }
};

/*!
\fn void calculoFiltroDiffGaussiano (Mat &DG, float s)
\brief Esta funcion calcula filtro unidimensional horizontal que codifica los
parametros de la derivada de una Gaussiana.
\param Mat &DG -> La matriz en donde se van a regresar los parámetros del filtro.
\param float s -> La desviación estandar que queremos tenga el filtro.
*/
void calculoFiltroDiffGaussiano (Mat &DG, float s)
{
    int filterSize, centerIdx, i, j;
    float K, s2, x;

    centerIdx = ceil(s * 3);
    filterSize = centerIdx * 2 + 1;
    DG = Mat::zeros(Size(filterSize, 1), CV_64FC1);
    K = 1./(sqrt(2*M_PI)*s*s*s);
    s2 = s * s;
    DG.at<double>(0,centerIdx) =  0;
    x = 1.f;
    for (j = centerIdx - 1, i = centerIdx + 1; i < DG.cols; ++i, --j, x+=1.f)
    {
        DG.at<double>(0,i) = - 0.5 * x * K * (exp(-0.5*x*x/s2)+exp(-0.5*pow(x+1.,2)/s2));
        DG.at<double>(0,j) = - DG.at<double>(0,i);
    }
}

/*!
\fn void findCorners (Mat &Ix, Mat Iy, Mat &C, int w, double k)
\brief  Esta funcion calcula la medida del detector de esquinas de Harris definida por la siguiente fórmula:
\f$C(G) = \det |G| - k trace^2(G)\f$,  donde \f$G=\left[\begin{array}{cc} \sum_{\tilde{x}} I_x^2(\tilde{x}) & \sum_{\tilde{x}} I_x(\tilde{x})I_y(\tilde{x})\\ \sum_{\tilde{x}} I_x(\tilde{x})I_y(\tilde{x}) &  \sum_{\tilde{x}}I_y^2(\tilde{x}) \end{array}\right]\f$ donde \f$\tilde{x} \in W(x)\f$ y \f$W(x)\f$ es una ventana de dimensiones \f$ w \times w\f$ centrada en la coordenada \f$x\f$.
 
 \param Mat &Ix La referencia a la matriz que contiene las derivadas horizaontales de la imagen a ser procesadas.
 \param Mat &Iy La referencia a la matriz que contiene las derivadas verticales de la imagen a ser procesadas.
 \param Mat &C La referencia a la matriz donde se almacenará el resultado
 \param int w El parámtro que define el tamaño de la ventana de cómputo.
 \param double k El parámtro que controla el comportamiento del detector de esquinas de Harris (usualmente un valor pequeño).
*/
void findCorners (Mat &Ix, Mat Iy, Mat &C, int w, double k)
{
    double g0, g1, g2;
    int i, j, x0, x1, y0, y1, w2;
    if (Ix.cols != Iy.cols || Ix.rows != Iy.rows)
        return;

    if (! (w % 2)) //Validamos que el w sea un número impar.
        w++;
    w2 = (w-1) / 2;

    Size sz(Ix.cols, Ix.rows);
    C = Mat::zeros(sz, CV_64FC1);

    x0 = y0 = w2;
    x1 = Ix.cols - w2;
    y1 = Iy.rows- w2;

    double *ptrC;
    Point p(1,0);
    for (i = y0; i < y1; ++i)
    {
        ptrC = C.ptr<double>(i) + x0;
        Rect roi(x0 - w2, i - w2, w, w);
        for (j = x0; j < x1; ++j, ptrC++, roi += p)
        {
            Mat ROIx = Ix(roi);
            Mat ROIy = Iy(roi);
            
            Mat_<double>::iterator itx, ity, end;

            itx = ROIx.begin<double>();
            ity = ROIy.begin<double>();
            end = ROIx.end<double>();
            g0 = g1 = g2 = 0;
            for (;itx != end; itx++, ity++)
            {
                g0 += *itx * *itx;
                g1 += *ity * *ity;
                g2 += *itx * *ity;
            }
            *ptrC = (g0*g1-g2*g2) - k * pow((g0+g1),2);
        }
    }
}

/*!
\fn void stdChange(int pos, void *data)
\brief Esta funcion modifica el valor de la desviación estandar y los filtros
almacenados en (filterData *)data. Esta funcion es invocada indirectamente por
el usuario al interactuar con la barra deslizante.
\param int pos  ->  La posicion de la barra deslizante al momento de ser invocada.
\param void *data  ->  Apuntador genérico a los datos.
*/
void stdChange(int pos, void *data)
{
    filterData *fd = (filterData *)data;

    //Calculamos el valor de fd->s en términos de la posicion de la barra.
    fd->s = (pos + 0.5) * fd->sInc;

    //Calculamos los filtros unidimensionales de la derivada de la Gaussiana,
    //(Horizontal y Vertical).
    calculoFiltroDiffGaussiano (fd->DGh, fd->s);
    fd->DGv = fd->DGh.clone();
    fd->DGv=fd->DGv.t();
    
    cout << "La desviación estandar es igual a: " << fd->s << endl;
}

/*!
\fn void kChange(int pos, void *data)
\brief Esta funcion modifica el valor del parámetro k, almacenado en una estructura del tipo HarrisData. Esta funcion es invocada indirectamente por
el usuario al interactuar con la barra deslizante.
\param int pos  ->  La posicion de la barra deslizante al momento de ser invocada.
\param void *data  ->  Apuntador genérico a los datos.
*/
void kChange(int pos, void *data)
{
    HarrisData *H = (HarrisData *)data;

    H->k = (pos + 0.5) * H->kInc;
    cout << "El parámetro k es igual a: " << H->k << endl;
    cout.flush();
}


/*!
\fn void wChange(int pos, void *data)
\brief Esta funcion modifica el valor del parámetro w, almacenado en una estructura del tipo HarrisData. Esta funcion es invocada indirectamente por el usuario al interactuar con la barra deslizante.
\param int pos  ->  La posicion de la barra deslizante al momento de ser invocada.
\param void *data  ->  Apuntador genérico a los datos.
*/
void wChange(int pos, void *data)
{
    ((HarrisData *)data)->W = pos;
}

int main(int argc, char **argv)
{
    Mat frame, grayFrame, grayFrameF, qframe, mask;
    Mat Outh, Outv, Corners, Out;
    double scaleX, scaleY;
    String inputName;
    bool first;
    filterData fd;
    HarrisData hd;
    int slidePosS = (int)(1.0 / fd.sInc);
    int slidePosK = (int)(0.08 / hd.kInc);
    int slidePosW = 5;
    struct timespec timeS, timeE;
    double endns, startns, timeMTransf;

    VideoCapture cap;

    if (argc < 2)
    {
        cerr << "Faltan Parámetros." << endl;
        cerr << "Uso: DeteccionBordes [NoCamara,NombreArchivo] " << endl << endl;
        exit(1);
    }
    
    
    if (isdigit(argv[1][0]))
       cap.open(atoi(argv[1]));  //Abre la camara determinada por el primer parámetro que se le paso al programa 
   else
      cap.open(argv[1]);

    if(!cap.isOpened())  // Si no se tuvo éxito, termina el programa.
        return -1;

    Mosaic M(Size(IM_WIDTH, IM_HEIGHT), 2, 2, 8, 8);

    //Definimos el tamaño de las imagenes a ser capturadas.
    cap.set(CAP_PROP_FRAME_WIDTH, IM_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, IM_HEIGHT);

    //Encontramos la razon entre el tamaño de imagenes que queremos y lo
    //que obtenemos en realidad. Esto es necesario para el caso que no
    //se pueda modificar el tamaño de la imagen de captura. Estas razones 
    //servirán mas adelante para escalar el tamaño de las imagenes adecuadas.
    scaleX = (double)IM_WIDTH / cap.get(CAP_PROP_FRAME_WIDTH);
    scaleY = (double)IM_HEIGHT / cap.get(CAP_PROP_FRAME_HEIGHT);

    

    //Abrimos las ventanas para mostrar los resultados.
    namedWindow( "Mosaico", 1 );

    //Invocamos de manera manual la funcion stdChange, para inicializar filtros.
    stdChange(slidePosS, (void *)&fd);

    //Creamos la barra deslizante con la que el usuario controlará el valor de la desviacion estandar y la asignamos a la ventana "Mosaico".
    createTrackbar("std", "Mosaico", &slidePosS, S_SLIDE_MAX, stdChange, (void *)&fd);

    //Invocamos de manera manual la funcion kChange.
    kChange(slidePosK, (void *)&hd);
    //Creamos la barra deslizante con la que el usuario controlará el valor del parámetro  k yla asignamos a la ventana "Mosaico".
    createTrackbar("k", "Mosaico", &slidePosK, K_SLIDE_MAX, kChange, (void *)&hd);

    //Invocamos de manera manual la funcion wChange.
    wChange(slidePosW, (void *)&hd);
    //Creamos la barra deslizante con la que el usuario controlará el valor del parámetro W  y la asignamos a la ventana "Mosaico".
    createTrackbar("w", "Mosaico", &slidePosW, W_SLIDE_MAX, wChange, (void *)&hd);



    first = true;
    for (int cont=0;;cont++)
    {
         //Capturamos una imagen, y validamos que haya funcionado la operacion.
        cap >> frame;
            

        //En la primera iteración inicializamos las imagenes que usaremos para
        //almacenar resultados.
        if (first)
        {
            Size sz(IM_WIDTH, IM_HEIGHT);

            Outh= Mat::zeros(sz, CV_64FC1);
            Outv = Mat::zeros(sz, CV_64FC1);
            Corners = Mat::zeros(sz, CV_64FC1);
            qframe = Mat::zeros(sz, frame.type());
            first = false;
        }

        if (!frame.empty())
        {

            //en dado caso que las imagenes capturado no sean del tamaño deseado
            //cambia el tamaño. y convierte la imagen de color a una imagen en tonos
            //de gris.
             if (scaleX != 1. || scaleY != 1.)
             {
                resize(frame, qframe, Size(), scaleX, scaleY, INTER_AREA);
                cvtColor (qframe, grayFrame, COLOR_BGR2GRAY);
             }
             else
                cvtColor (frame, grayFrame, COLOR_BGR2GRAY);
        }

        //Despliega la imagen capturada en una ventana y conviertela a 
        //una respresentacion de dobles.
        M.setFigure(grayFrame, 0, 0);
        grayFrame.convertTo (grayFrameF, CV_64FC1);


        //Aplica filtros de la primera derivada de la Gausssiano, y calcula su
        //representacion en terminos de magnitud y orientación  (fase).  Normaliza
        //los resultados, y convierte a una imagen CV_8UC1 y despliega.
        filter2D (grayFrameF, Outh, -1, fd.DGh);
        filter2D (grayFrameF, Outv, -1, fd.DGv);


        clock_gettime (CLOCK_REALTIME, &timeS);
        findCorners (Outh, Outv, Corners, hd.W, hd.k);
        clock_gettime (CLOCK_REALTIME, &timeE);
        startns = (double)(timeS.tv_sec)*1e9 + (double)(timeS.tv_nsec);
        endns = (double)(timeE.tv_sec)*1e9 + (double)(timeE.tv_nsec);
        timeMTransf = (endns -startns)/10;
        cout<< "Tiempo: " << timeMTransf << " nano-segundos" << endl;
        cout.flush();

        normalize(Outh, Outh, 0, 255, NORM_MINMAX);
        normalize(Outv, Outv, 0, 255, NORM_MINMAX);
        //log(Corners, Corners);
        normalize(Corners, Corners, 0, 255, NORM_MINMAX);
        Outh.convertTo (Out, CV_8UC1);
        M.setFigure(Out, 0, 1);
        Outv.convertTo (Out, CV_8UC1);
        M.setFigure(Out, 1, 0);
        Corners.convertTo (Out, CV_8UC1);
        M.setFigure(Out, 1, 1);

        M.show("Mosaico");

        //Si el usuario oprime una tecla, termina el ciclo.
        if (waitKeyEx( 30 ) >= 0 )
            break;
          
    }

    //Cierra ventanas que fueron abiertas.
    destroyWindow ("Mosaico");

    return 0;
}
