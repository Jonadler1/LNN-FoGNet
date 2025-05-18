# LNN-FoGNet
# LNN-FoGNet – Liquid Neural Networks for Real-Time Freezing-of-Gait Detection

<div align="center">
<img src="https://img.shields.io/badge/python-3.9-blue" />
<img src="https://img.shields.io/badge/tensorflow-2.18-important" />
<img src="https://img.shields.io/badge/keras-3.6-important" />
<img src="https://img.shields.io/badge/license-MIT-green" />
</div>

## ✨ Project summary
Freezing-of-Gait (FoG) is one of the most disabling symptoms of Parkinson’s disease.  
**LNN-FoGNet** explores whether **Liquid Neural Networks (LNNs)**—recurrent models whose neurons adapt their own time-constants—can deliver **LSTM-grade accuracy** while remaining **smaller, faster, and more energy-efficient** for round-the-clock wearables.

* **Dataset** [Daphnet FoG](https://archive.ics.uci.edu/ml/datasets/daphnet+freezing+of+gait).  
* **Models compared** Liquid Neural Network (LNN / LTC)  |  Long Short-Term Memory (LSTM)  |  Continuous-Time RNN (CTRNN).  
* **Key results**  
  * Mean F1 ≈ 0.95 on 5-fold subject-wise CV.  
  * LNN converges in **≈½ the epochs and 1/10th the training time** of LSTM.  
  * Inference latency per step is **tens-of-times faster** than LSTM on the same hardware.

For a full scientific write-up see **JA_Final_Project_Report_2025_DY.docx** in this repository.

---

## 🗂️ Repository layout

