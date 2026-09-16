from ultralytics import YOLO

modelo = YOLO("yolov8n.pt")

resultados = modelo.predict(
   source= "estacionamento.jpg",
   conf=0.5,
   #conf=0.25
   classes=[2],
   save=True
)

deteccoes = resultados[0].boxes
print("Número de carros detectados:", len(deteccoes))
print("Confianças individuais:", deteccoes.conf.tolist())
print("Confiança média:", deteccoes.conf.mean().item())
