import sys
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtMultimedia import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimediaWidgets import *
from Facial.reFacial_class import ReconocimientoFacial
from Facial.capturaRostros_class import CapturaFacial
from Facial.capturaVideo import CapturaFacialVideo
from Facial.entrenadorFacial_class import EntrenadorFacial
from Color.color_class import ReidentificacionColor
from Color.captura_class import CapturaColor
from Color.entrenadorColor_class import EntrenadorColor
from Textura.textura_class import ReidentificacionTextura
from Textura.capturaTextura_class import CapturaTextura
from Textura.entrenadorTextura_class import EntrenadorTextura
from Silueta.silueta_class import ReidentificacionSilueta
from Silueta.capturaSilueta_class import CapturaSilueta
from Silueta.entrenadorSilueta_class import EntrenadorSilueta
import cv2
from time import sleep
 

class VentanaPrincipal(QMainWindow) :
    def __init__(self):
        super(VentanaPrincipal, self).__init__()
        loadUi ("interfaz_reid.ui", self)
        
        #variables de la interfaz
        self.camara=False
        self.filepath=""
        self.menu_index=4
        self.maximizar=False
        self.playing_video=False
        self.capturing_camera=False
        self.training=True
        self.index_rb=0
        self.capturing_video=False
        self.count_capturing=0
        # ocultamos los botones
        self.bt_restaurar.hide ()
        self.bt_stop_captura.hide()
        self.bt_entrenamiento.setEnabled(False)
        #control barra de titulos
        self.bt_minimizar.clicked.connect(self.control_bt_minimizar)
        self.bt_restaurar.clicked.connect(self.control_bt_normal)
        self.bt_maximizar.clicked.connect(self.control_bt_maximizar)
        self.bt_cerrar.clicked.connect(lambda : self.close())
        #eliminar barra y de titulo opacidad
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setWindowOpacity(1)
        #SizeGrip redimensionar la interfaz gráfica
        self.gripSize=10
        self.grip = QtWidgets.QSizeGrip(self)
        self.grip.resize(self.gripSize ,self.gripSize)
        #mover ventana
        self.frame_superior.mouseMoveEvent =self.mover_ventana
        #acceder a las paginas
        self.bt_color.clicked.connect(self.pagina_uno)
        self.bt_facial.clicked.connect(self. pagina_dos)
        self.bt_silueta.clicked.connect(self.pagina_tres)
        self.bt_textura.clicked.connect(self.pagina_cuatro)
        self.bt_entrenamiento.clicked.connect(self.pagina_cinco)
        self.bt_next_one.clicked.connect(self.paso_uno_next)
        self.bt_next_two.clicked.connect(self.paso_dos_next)
        self.bt_after_two.clicked.connect(self.paso_dos_after)
        self.bt_next_three.clicked.connect(self.paso_tres_next)
        self.bt_after_three.clicked.connect(self.paso_tres_after)
        self.bt_after_four.clicked.connect(self.paso_cuatro_after)
        self.bt_end.clicked.connect(self.paso_cuatro_end)
        #video connections
        self.bt_select_video.clicked.connect(self.abrir_video)
        self.bt_play.clicked.connect(self.start_video)
        self.bt_stop.clicked.connect(self.stop_video)
        self.bt_camara.clicked.connect(self.start_camera)
        self.bt_desac_camara.clicked.connect(self.stop_camera)
        self.bt_desac_camara.hide ()
        self.bt_stop.hide()
        self.bt_camara.setEnabled(False)
        self.bt_play.setEnabled(False)
        #Pasos buttons
        self.bt_cap_video.clicked.connect(self.captura_video)
        self.bt_stop_captura.clicked.connect(self.stop_cap_video)
        self.bt_cap_camara.clicked.connect(self.captura_camara)
        self.bt_entrenar_modelo.clicked.connect(self.entrenar_modelo)
        self.bt_end.setEnabled(False)
        self.label_23.hide()
        #set size of frames
        self.label_video_color.setFixedWidth(404)
        self.frame_45.setFixedWidth(440)
        self.label_video_facial.setFixedWidth(404)
        self.frame_25.setFixedWidth(450)
        self.label_video_textura.setFixedWidth(404)
        self.frame_10.setFixedWidth(440)
        self.label_video_silueta.setFixedWidth(404)     
        self.frame_35.setFixedWidth(450)
        self.frame_53.setFixedWidth(550)
        
       
    #Control de las ventanas   
    def control_bt_minimizar(self):
        self.showMinimized()
    def control_bt_normal(self) :
        self.frame_53.setFixedWidth(550)
        
        if self.camara==False:
            self.label_video_entrada.clear()
            self.label_video_color.setFixedWidth(404)
            self.frame_45.setFixedWidth(450)
            self.label_video_facial.setFixedWidth(404)
            self.frame_25.setFixedWidth(450)
            self.frame_35.setFixedWidth(440)
            self.label_video_silueta.setFixedWidth(404)
            self.label_video_textura.setFixedWidth(404)
            self.frame_10.setFixedWidth(450)
        else:
            self.label_video_color.setFixedWidth(404)
            self.frame_45.setFixedWidth(440)
            self.label_video_facial.setFixedWidth(404)
            self.frame_25.setFixedWidth(450)
            self.label_video_textura.setFixedWidth(404)
            self.frame_10.setFixedWidth(440)
            self.label_video_silueta.setFixedWidth(404)     
            self.frame_35.setFixedWidth(400)
        if self.training:
            if self.index_rb==0:
                self.frame_5.setFixedWidth(410)
                self.label_video_entrada.setFixedWidth(390)
            if self.index_rb==1:
                self.frame_5.setFixedWidth(425)
                self.label_video_entrada.setFixedWidth(400)
            if self.index_rb==2:
                self.frame_5.setFixedWidth(410)
                self.label_video_entrada.setFixedWidth(390)
            if self.index_rb==3:
                self.frame_5.setFixedWidth(410)
                self.label_video_entrada.setFixedWidth(390)
                
        else:
            self.frame_5.setFixedWidth(425)   
        self.label_video_color.clear()
        self.label_video_facial.clear()
        self.label_video_silueta.clear()
        self.label_video_textura.clear()
        self.maximizar=False
        
        self.showNormal()
        self.bt_restaurar.hide()
        self.bt_maximizar.show()
    def control_bt_maximizar(self):
        if self.camara==False:
            self.label_video_entrada.clear()
        if self.camara:
            self.frame_35.setFixedWidth(620)
            self.frame_45.setFixedWidth(620)
            self.frame_25.setFixedWidth(620)
            self.frame_10.setFixedWidth(620)
            self.frame_35.setFixedWidth(590)
            self.label_video_color.setFixedWidth(650)
            self.label_video_facial.setFixedWidth(650)
            self.label_video_silueta.setFixedWidth(750)
            self.label_video_textura.setFixedWidth(650)
        else:
            self.label_video_color.setFixedWidth(750)
            self.frame_45.setFixedWidth(740)
            self.label_video_textura.setFixedWidth(750)
            self.frame_10.setFixedWidth(740)
            self.label_video_facial.setFixedWidth(750)
            self.frame_25.setFixedWidth(740)
            self.label_video_silueta.setFixedWidth(750)
            self.frame_35.setFixedWidth(740)
            
        self.label_video_color.clear()
        self.label_video_facial.clear()
        self.label_video_silueta.clear()
        self.label_video_textura.clear()        
        if self.training:
            if self.capturing_video:
                self.frame_5.setFixedWidth(770)
                self.label_video_entrada.setFixedWidth(750)
            else:
                if self.index_rb==0:
                    self.frame_5.setFixedWidth(585)
                    self.label_video_entrada.setFixedWidth(560)
                if self.index_rb==1:
                    self.frame_5.setFixedWidth(600)
                    self.label_video_entrada.setFixedWidth(580)
                if self.index_rb==2:
                    self.frame_5.setFixedWidth(585)
                    self.label_video_entrada.setFixedWidth(560)
                if self.index_rb==3:
                    self.frame_5.setFixedWidth(585)
                    self.label_video_entrada.setFixedWidth(560)
        else:
            self.frame_5.setFixedWidth(770)
            self.label_video_entrada.setFixedWidth(750) 
          
        self.frame_53.setFixedWidth(700)
        self.showMaximized()
        self.bt_maximizar.hide ()
        self.bt_restaurar.show ()
        # creamos el metodo sombra

    ## SizeGrip redimensionar la interfaz
    def resizeEvent(self, event):
        rect= self. rect ()
        self.grip.move(rect.right() - self.gripSize, rect.bottom () - self.gripSize)

    ## mover ventana
    def mousePressEvent (self, event) :
        self.clickPosition = event.globalPos ()
    def mover_ventana (self, event):
        if self.isMaximized () == False :
            if event.buttons () == QtCore.Qt.LeftButton :
                self.move(self.pos()+ event.globalPos() -self.clickPosition)
                self.clickPosition =event.globalPos()
                event.accept ()
            if event.globalPos().y() <=10:
                self.showMaximized ()
                self.bt_maximizar.hide ()
                self.bt_restaurar.show()
            else:
                self.showNormal() 
                self.bt_restaurar.hide()
                self.bt_maximizar.show()
        # Metodo para mover el menu
    #Cambiar con el boton de menú las páginas
    def pagina_uno(self):
        self.capturing_camera=False
        self.capturing_video=False
        self.training=False
        if self.isMaximized():
            self.frame_5.setFixedWidth(770)
            self.label_video_entrada.setFixedWidth(750) 
        self.bt_color.setEnabled(False)
        self.bt_facial.setEnabled(True)
        self.bt_silueta.setEnabled(True)
        self.bt_textura.setEnabled(True)
        self.bt_entrenamiento.setEnabled(True)
        self.bt_camara.setEnabled(True)
        self.bt_play.setEnabled(True)
        self.stackedWidget.setCurrentWidget(self.page_color)
        if self.playing_video:
            self.stop_video()
        if self.camara:
            self.stop_camera()
        self.menu_index=0    
    def pagina_dos(self):
        self.capturing_camera=False
        self.capturing_video=False
        if self.isMaximized():
            self.frame_5.setFixedWidth(770)
            self.label_video_entrada.setFixedWidth(750) 
        self.training=False
        self.bt_facial.setEnabled(False)
        self.bt_color.setEnabled(True)
        self.bt_silueta.setEnabled(True)
        self.bt_textura.setEnabled(True)
        self.bt_entrenamiento.setEnabled(True)
        self.bt_camara.setEnabled(True)
        self.bt_play.setEnabled(True)
        self.stackedWidget.setCurrentWidget(self.page_facial)
        if self.playing_video:
            self.stop_video()
        if self.camara:
            self.stop_camera()
        self.menu_index=1
    def pagina_tres(self):
        self.capturing_camera=False
        self.capturing_video=False
        if self.isMaximized():
            self.frame_5.setFixedWidth(770)
            self.label_video_entrada.setFixedWidth(750) 
        self.training=False
        self.bt_silueta.setEnabled(False)
        self.bt_facial.setEnabled(True)
        self.bt_color.setEnabled(True)
        self.bt_textura.setEnabled(True)
        self.bt_entrenamiento.setEnabled(True)
        self.bt_camara.setEnabled(True)
        self.bt_play.setEnabled(True)
        self.stackedWidget.setCurrentWidget(self.page_silueta)
        if self.playing_video:
            self.stop_video()
        if self.camara:
            self.stop_camera()
        self.menu_index=2
    def pagina_cuatro(self):
        self.capturing_camera=False
        self.capturing_video=False
        if self.isMaximized():
            self.frame_5.setFixedWidth(770)
            self.label_video_entrada.setFixedWidth(750) 
        self.training=False
        self.bt_textura.setEnabled(False)
        self.bt_facial.setEnabled(True)
        self.bt_silueta.setEnabled(True)
        self.bt_color.setEnabled(True)
        self.bt_entrenamiento.setEnabled(True)
        self.bt_camara.setEnabled(True)
        self.bt_play.setEnabled(True)
        self.stackedWidget.setCurrentWidget(self.page_textura)
        if self.playing_video:
            self.stop_video()
        if self.camara:
            self.stop_camera()
        self.menu_index=3
    def pagina_cinco(self):
        
        if self.playing_video:
            self.stop_video()
        if self.camara:
            self.stop_camera()
        self.training=True
        self.menu_index=4
        self.bt_play.setEnabled(False)
        self.bt_textura.setEnabled(True)
        self.bt_facial.setEnabled(True)
        self.bt_silueta.setEnabled(True)
        self.bt_color.setEnabled(True)
        self.bt_camara.setEnabled(False)
        self.bt_entrenamiento.setEnabled(False)
        self.stackedWidget.setCurrentWidget(self.page_entrenamiento)
    def paso_uno_next(self):
        self.name= self.lineEdit.text()
        if self.name=="":
            self.label_paso_uno.setText("Ingresa un nombre de por lo menos 5 caracteres")
        else:
            self.stackedWidget_pasos.setCurrentWidget(self.pagina_paso_dos)
    def paso_dos_next(self):
        if self.rb_color.isChecked():
            self.stackedWidget_pasos.setCurrentWidget(self.pagina_paso_tres)
            self.index_rb=0
        if self.rb_facial.isChecked():
            self.stackedWidget_pasos.setCurrentWidget(self.pagina_paso_tres)
            self.index_rb=1
        if self.rb_silueta.isChecked():
            self.stackedWidget_pasos.setCurrentWidget(self.pagina_paso_tres)
            self.index_rb=2
        if self.rb_textura.isChecked():
            self.stackedWidget_pasos.setCurrentWidget(self.pagina_paso_tres)
            self.index_rb=3
        else:
            self.label_paso_dos.setText("Seleccione una opción")
        
    def paso_dos_after(self):
        self.label_paso_uno.setText("")
        self.stackedWidget_pasos.setCurrentWidget(self.pagina_paso_uno)
    def paso_tres_next(self):
        if self.count_capturing>0:
            self.stackedWidget_pasos.setCurrentWidget(self.pagina_paso_cuatro)
            if self.index_rb==0:
                self.label_22.setText("5. Selecciona Color en el menú principal o presiona Finalizar.")
            if self.index_rb==1:
                self.label_22.setText("5. Selecciona Facial en el menú principal o presiona Finalizar.")
            if self.index_rb==2:
                self.label_22.setText("5. Selecciona Silueta en el menú principal o presiona Finalizar.")
            if self.index_rb==3:
                self.label_22.setText("5. Selecciona Textura en el menú principal o presiona Finalizar.")
            
        else:
            self.label_18.hide()
            self.label_23.show()
    def paso_tres_after(self):
        self.label_paso_dos.setText("")
        self.stackedWidget_pasos.setCurrentWidget(self.pagina_paso_dos)
    def paso_cuatro_after(self):
        self.label_18.show()
        self.label_23.hide()
        self.stackedWidget_pasos.setCurrentWidget(self.pagina_paso_tres)
    def paso_cuatro_end(self):
        if self.index_rb==0:
            self.pagina_uno()
        if self.index_rb==1:
            self.pagina_dos()
        if self.index_rb==2:
            self.pagina_tres()
        if self.index_rb==3:
            self.pagina_cuatro()
    def captura_video(self):
        
        self.capturing_video=True
        self.label_video_entrada.clear()
        if self.isMaximized():
            self.frame_5.setFixedWidth(770)
            self.label_video_entrada.setFixedWidth(750)
            
        if self.filepath!="":
            self.count_capturing+=1
            self.bt_stop_captura.show()
            self.bt_cap_video.hide()
            self.bt_next_three.setEnabled(False)
            self.bt_after_three.setEnabled(False)
            self.bt_select_video.setEnabled(False)
            self.bt_color.setEnabled(False)
            self.bt_facial.setEnabled(False)
            self.bt_silueta.setEnabled(False)
            self.bt_textura.setEnabled(False)
            self.bt_cap_camara.setEnabled(False)
            if self.index_rb==0:
                self.cap_video=CapturaColor(self.name, self.filepath, self.label_video_entrada)
            if self.index_rb==1:
                self.cap_video=CapturaFacialVideo(self.name, self.filepath, self.label_video_entrada)
            if self.index_rb==2:
                self.cap_video=CapturaSilueta(self.name, self.filepath, self.label_video_entrada)
            if self.index_rb==3:
                self.cap_video=CapturaTextura(self.name, self.filepath, self.label_video_entrada)
            self.cap_video.start()
            self.cap_video.Image_salida_upd.connect(self.cap_video_upd_slot)
        else:
            self.label_video_entrada.setText("Seleccione un video")
            self.label_video_entrada.setAlignment(QtCore.Qt.AlignCenter)
    def cap_video_upd_slot(self,Image):
        self.label_video_entrada.setPixmap(QPixmap.fromImage(Image))
    def stop_cap_video(self):
        self.bt_stop_captura.hide()
        if self.capturing_camera:
            self.bt_cap_camara.show()
            self.bt_cap_video.setEnabled(True)
        else:
            self.bt_cap_video.show()
            self.bt_cap_camara.setEnabled(True)
        self.capturing_camera=False
        self.capturing_video=False
        self.bt_after_three.setEnabled(True)
        self.bt_next_three.setEnabled(True)
        self.bt_select_video.setEnabled(True)
        self.bt_color.setEnabled(True)
        self.bt_facial.setEnabled(True)
        self.bt_silueta.setEnabled(True)
        self.bt_textura.setEnabled(True)
        self.cap_video.stop()
    #Captura cámara
    def captura_camara(self):
        self.count_capturing+=1
        self.capturing_video=False
        self.label_video_entrada.clear()
        self.capturing_camera=True
        self.bt_stop_captura.show()
        self.bt_cap_camara.hide()
        self.bt_next_three.setEnabled(False)
        self.bt_after_three.setEnabled(False)
        self.bt_select_video.setEnabled(False)
        self.bt_cap_video.setEnabled(False)
        self.bt_color.setEnabled(False)
        self.bt_facial.setEnabled(False)
        self.bt_silueta.setEnabled(False)
        self.bt_textura.setEnabled(False)
        if self.index_rb==0:
            self.cap_video=CapturaColor(self.name, 0, self.label_video_entrada)
            if self.isMaximized ():
                self.frame_5.setFixedWidth(585)
                self.label_video_entrada.setFixedWidth(560)
            else:
                self.frame_5.setFixedWidth(410)                
        if self.index_rb==1:
            self.cap_video=CapturaFacial(self.name, 0, self.label_video_entrada)
            if self.isMaximized ():
                self.frame_5.setFixedWidth(600)
                self.label_video_entrada.setFixedWidth(580)
            else:
                self.frame_5.setFixedWidth(425)
        if self.index_rb==2:
            self.cap_video=CapturaSilueta(self.name, 0, self.label_video_entrada)
            if self.isMaximized ():
                self.frame_5.setFixedWidth(585)
                self.label_video_entrada.setFixedWidth(560)
            else:
                self.frame_5.setFixedWidth(410)
        if self.index_rb==3:
            self.cap_video=CapturaTextura(self.name, 0, self.label_video_entrada)
            if self.isMaximized ():
                self.frame_5.setFixedWidth(585)
                self.label_video_entrada.setFixedWidth(560)
            else:
                self.frame_5.setFixedWidth(410)
        self.cap_video.start()
        self.cap_video.Image_salida_upd.connect(self.cap_video_upd_slot)
    #entrenar modelo
    def entrenar_modelo(self):
        if self.index_rb==0:
            self.entrenador=EntrenadorColor(self.bt_entrenar_modelo, self.bt_end)
        if self.index_rb==1:
            self.entrenador=EntrenadorFacial(self.bt_entrenar_modelo, self.bt_end)
        if self.index_rb==2:
            self.entrenador=EntrenadorSilueta(self.bt_entrenar_modelo, self.bt_end)
        if self.index_rb==3:
            self.entrenador=EntrenadorTextura(self.bt_entrenar_modelo, self.bt_end)
        self.entrenador.start()
            
    #Abrir video
    def abrir_video(self):
        path=QFileDialog.getOpenFileName(self, "Abrir", "/")
        self.filepath=path[0]
        if self.filepath=="":
            self.bt_cap_video.setEnabled(False)
            return
        else:
            self.bt_cap_video.setEnabled(True)
            
    #camera funcion
    def start_camera(self):
        self.camara=True
        self.bt_play.setEnabled(False)
        self.bt_select_video.setEnabled(False)
        self.bt_camara.hide()
        self.bt_desac_camara.show()
        self.label_video_entrada.setText("La cámara esta siendo usada en la otra instacia")
        self.label_video_entrada.setAlignment(QtCore.Qt.AlignCenter)
        if self.menu_index==0:
            self.video_salida_camara=ReidentificacionColor(self.label_video_color, 0)
            if self.isMaximized () == True :
                self.frame_45.setFixedWidth(620)
                self.label_video_color.setFixedWidth(750)                
        if self.menu_index==1:
            self.video_salida_camara=ReconocimientoFacial(self.label_video_facial, 0)
            if self.isMaximized () == True :
                self.frame_25.setFixedWidth(620)
                self.label_video_facial.setFixedWidth(590)
        if self.menu_index==2:
            self.video_salida_camara=ReidentificacionSilueta(self.label_video_silueta,0)
            if self.isMaximized () == True :
                self.frame_35.setFixedWidth(590)
                self.label_video_silueta.setFixedWidth(750)
            else:
                self.frame_35.setFixedWidth(410)
                self.label_video_silueta.setFixedWidth(420)
        if self.menu_index==3:
            self.video_salida_camara=ReidentificacionTextura(self.label_video_textura, 0)
            if self.isMaximized () == True :
                self.frame_10.setFixedWidth(620)
                self.label_video_textura.setFixedWidth(750)
        self.video_salida_camara.start()
        self.video_salida_camara.Image_salida_upd.connect(self.Image_salida_upd_slot)
    def stop_camera(self):
        self.camara=False
        self.bt_play.setEnabled(True)
        self.bt_select_video.setEnabled(True)
        self.label_video_entrada.clear()
        self.bt_desac_camara.hide()
        self.bt_camara.show()
        self.label_video_color.clear()
        self.label_video_facial.clear()
        self.label_video_silueta.clear()
        self.label_video_textura.clear()
        self.video_salida_camara.stop()
    def start_video_salida(self):
        if self.menu_index==0:
            if self.camara==False:
                self.video_salida=ReidentificacionColor(self.label_video_color, self.filepath)
                if self.isMaximized () == True :
                    self.label_video_color.setFixedWidth(740)
                    self.frame_45.setFixedWidth(740)
                else:
                    self.label_video_color.setFixedWidth(404)
                    self.frame_45.setFixedWidth(450)                    
        if self.menu_index==1:
            if self.camara==False:
                self.video_salida=ReconocimientoFacial(self.label_video_facial, self.filepath)
                if self.isMaximized () == True :
                    self.label_video_facial.setFixedWidth(750)
                    self.frame_25.setFixedWidth(740)
                else:
                    self.label_video_facial.setFixedWidth(404)
                    self.frame_25.setFixedWidth(450)
        if self.menu_index==2:
            if self.camara==False:
                self.video_salida=ReidentificacionSilueta(self.label_video_silueta,self.filepath)
                if self.isMaximized () == True :
                    self.label_video_silueta.setFixedWidth(750)
                    self.frame_35.setFixedWidth(740)
                else:
                    self.frame_35.setFixedWidth(440)
                    self.label_video_silueta.setFixedWidth(404)                
        if self.menu_index==3:
            self.video_salida=ReidentificacionTextura(self.label_video_textura, self.filepath)
            if self.isMaximized () == True :
                self.label_video_textura.setFixedWidth(750)
                self.frame_10.setFixedWidth(740)
            else:
                self.label_video_textura.setFixedWidth(404)
                self.frame_10.setFixedWidth(450)
        self.video_salida.start()
        self.video_salida.Image_salida_upd.connect(self.Image_salida_upd_slot)
    def Image_salida_upd_slot(self,Image):
        if self.menu_index==0:
            self.label_video_color.setPixmap(QPixmap.fromImage(Image))
        if self.menu_index==1:
            self.label_video_facial.setPixmap(QPixmap.fromImage(Image))
        if self.menu_index==2:
            self.label_video_silueta.setPixmap(QPixmap.fromImage(Image))
        if self.menu_index==3:
            self.label_video_textura.setPixmap(QPixmap.fromImage(Image))
    def start_video(self):
        if self.filepath=="":
            self.label_video_entrada.setText("Seleccione un video")
            self.label_video_entrada.setAlignment(QtCore.Qt.AlignCenter)
        else:
            if self.isMaximized ():
                self.frame_5.setFixedWidth(770)
            self.camara=False
            self.playing_video=True
            self.bt_camara.setEnabled(False)
            self.bt_play.hide()
            self.bt_stop.show()
            self.video_entrada=video_entrada(self.label_video_entrada, self.filepath, self.bt_play, self.bt_stop)
            self.video_entrada.start()
            self.video_entrada.Image_entrada_upd.connect(self.Image_entrada_upd_slot)
            if self.menu_index!=4:
                self.start_video_salida()
    def Image_entrada_upd_slot(self,Image):
        self.label_video_entrada.setPixmap(QPixmap.fromImage(Image)) 
    def stop_video(self):
        self.playing_video=False
        self.bt_camara.setEnabled(True)
        self.bt_stop.hide()
        self.bt_play.show()
        self.label_video_entrada.clear()
        self.label_video_color.clear()
        self.label_video_facial.clear()
        self.label_video_silueta.clear()
        self.label_video_textura.clear()
        self.video_entrada.stop()
        if self.menu_index!=4:
            self.video_salida.stop()
class video_entrada(QThread):
    def __init__(self, label_video_entrada, filepath,bt_play,bt_stop):
        super().__init__()
        self.label_video_entrada = label_video_entrada
        self.filepath=filepath
        self.bt_play=bt_play
        self.bt_stop=bt_stop
    ##579x434 maximized 304,267 minimzed 2x    
    Image_entrada_upd=pyqtSignal(QImage)
    def run(self):
        self.hilo_corriendo=True
        cap=cv2.VideoCapture(self.filepath)
        
        while self.hilo_corriendo:
            ret,frame=cap.read()
            if ret:
                Image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                convertir_QT=QImage(Image.data,Image.shape[1],Image.shape[0],QImage.Format_RGB888)
                pic=convertir_QT.scaled(self.label_video_entrada.width(),self.label_video_entrada.height(),Qt.KeepAspectRatio)
                sleep(0.06)
                self.Image_entrada_upd.emit(pic)
    def stop(self):
        self.hilo_corriendo=False
        self.quit()                        
    
if __name__=="__main__":
    app=QApplication(sys.argv)
    mi_app=VentanaPrincipal()
    mi_app.show()
    sys.exit(app.exec_())