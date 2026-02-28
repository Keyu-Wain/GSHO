# Main execution script
import pandas as pd
import joblib
from config import CONFIG
from data_loader import load_and_preprocess_data
from models import train_traditional_models, train_mlp_model


def main():

    X, y = load_and_preprocess_data(CONFIG)

    traditional_results = train_traditional_models(X, y, CONFIG)

    mlp_results, best_mlp_model = train_mlp_model(X, y, CONFIG)

    final_results = traditional_results
    if mlp_results:
        final_results["MLP"] = mlp_results

    summary_data = {}
    for model_name, res in final_results.items():
        summary_data[model_name] = {
            "Average R²": res["Average R²"],
            "Average RMSE": res["Average RMSE"],
            "Average MAE": res["Average MAE"]
        }

    summary_df = pd.DataFrame(summary_data).T.round(4)
    summary_df = summary_df.sort_values(by="Average R²", ascending=False)

    # Save results
    try:
        with pd.ExcelWriter(CONFIG["OUTPUT_EXCEL_PATH"]) as writer:
            summary_df.to_excel(writer, sheet_name="Metrics Summary")
            for model_name, res in final_results.items():
                if "Params Summary" in res:
                    res["Params Summary"].to_excel(writer, sheet_name=f"{model_name}_Params")
                else:
                    pd.DataFrame([res["Best Params"]]).to_excel(writer, sheet_name=f"{model_name}_Params")
    except Exception as e:
        print(f"Excel save error: {str(e)}")

if __name__ == "__main__":
    main()