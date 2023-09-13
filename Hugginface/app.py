import gradio as gr
import joblib as jb

def predict(sex, age, pclass):
    model = jb.load('model.pkl')
    pclass = int(pclass)
    p = model.predict_proba([[sex,age,pclass]])[0]

    return {"Não sobreviveu": p[0], "Sobreviveu": p[1]}

demo = gr.Interface(fn = predict,
                    inputs=[gr.Dropdown(choices=["male", "female"], type="index"),
                            "number",
                            gr.Dropdown(choices=["1","2","3"], type="value")],
                    outputs="label")
demo.launch()