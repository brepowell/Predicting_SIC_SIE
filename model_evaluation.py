if EVALUATING_ON:
    start_full_evaluation = time.perf_counter()
    # Create a string buffer to capture output
    captured_output = io.StringIO()
    
    # Redirect stdout to the buffer
    sys.stdout = captured_output

    # --- CALL METRIC FUNCTIONS HERE ---
    
    logging.info("\nCalling SIC plotting functions...")
    # 1. Calculate and Log Overall Spatial Errors (Overall SIC's MAE, MSE, RMSE)
    # This function calculates and prints overall spatial error metrics.
    logging.info("\nErrors Overall...")
    start_time_spatial_errors = time.perf_counter()
    if not sic_ds.empty:
        calculate_and_log_spatial_errors(sic_ds, title_suffix=" (Overall)")
    end_time_spatial_errors = time.perf_counter()
    print(f"Elapsed time for Overall Spatial Error Calculation: {end_time_spatial_errors - start_time_spatial_errors:.2f} seconds")

    # 2. Plot Temporal Degradation (SIC Over Each Forecast Day)
    # This function plots MAE and RMSE degradation over the forecast horizon for SIC.
    logging.info("\nTemporal Degradation (SIC) over forecast days ...")
    start_time_sic_temporal_degradation = time.perf_counter()
    if not sic_ds.empty:
        plot_SIC_temporal_degradation(sic_ds, model_version, patching_strategy_abbr)
    end_time_sic_temporal_degradation = time.perf_counter()
    print(f"Elapsed time for SIC Temporal Degradation Plot: {end_time_sic_temporal_degradation - start_time_sic_temporal_degradation:.2f} seconds")

    # 3. Plot Actual vs. Predicted SIC Distribution (Overall SIC)
    # This function plots overlapping histograms of actual vs. predicted SIC values.
    logging.info("\nDistance between actual and predicted ...")
    start_time_actual_vs_predicted_sic_dist = time.perf_counter()
    if not sic_ds.empty:
        plot_actual_vs_predicted_sic_distribution(sic_ds, model_version, patching_strategy_abbr, num_bins=50, title_suffix=" (Overall)")
    end_time_actual_vs_predicted_sic_dist = time.perf_counter()
    print(f"Elapsed time for Overall Actual vs Predicted SIC histogram: {end_time_actual_vs_predicted_sic_dist - start_time_actual_vs_predicted_sic_dist:.2f} seconds")

    logging.info("\nCalling SIE plotting functions...")
    # 4. Log Classification Report (SIE as a binary value of SIC with 15% threshold)
    # This function provides a classification report for Sea Ice Extent (SIE).
    logging.info("\nClassification Report for SIE ...")
    start_time_classification_report = time.perf_counter()
    if not sic_ds.empty:
        sie_threshold = 0.15 # Define sie_threshold
        log_classification_report(sic_ds, threshold=sie_threshold)
    end_time_classification_report = time.perf_counter()
    print(f"Elapsed time for Overall Classification Report: {end_time_classification_report - start_time_classification_report:.2f} seconds")
    
    # 5. Plot Overall SIE Confusion Matrix (SIE here is derived as a binary value of SIC with 15% threshold)
    # This function generates a confusion matrix plot for SIE classification.
    logging.info("\nConfusion Matrix for SIE ...")
    start_time_confusion_matrix = time.perf_counter()
    if not sic_ds.empty:
        plot_sie_confusion_matrix(sic_ds, threshold=sie_threshold, model_version=model_version, patching_strategy_abbr=patching_strategy_abbr, forecast_day=None) # None for overall
    end_time_confusion_matrix = time.perf_counter()
    print(f"Elapsed time for Overall Confusion Matrix Plot: {end_time_confusion_matrix - start_time_confusion_matrix:.2f} seconds")
    
    # 6. Plot Overall ROC Curve (SIE here is derived as a binary value of SIC with 15% threshold)
    # This function plots the Receiver Operating Characteristic (ROC) curve and calculates AUC.
    logging.info("\nROC Curve for SIE ...")
    start_time_roc_curve = time.perf_counter()
    if not sic_ds.empty:
        plot_roc_curve(sic_ds, model_version=model_version, patching_strategy_abbr=patching_strategy_abbr, threshold=sie_threshold, forecast_day=None) # None for overall
    end_time_roc_curve = time.perf_counter()
    print(f"Elapsed time for Overall ROC Curve Plot: {end_time_roc_curve - start_time_roc_curve:.2f} seconds")

    # 7. Plot F1-Score Degradation (SIE here is derived as a binary value of SIC with 15% threshold)
    # This function plots the F1-score degradation for SIE classification.
    logging.info("\nF1 Score for SIE ...")
    start_time_f1_degradation = time.perf_counter()
    if not sic_ds.empty: # F1-score uses the same SIC temporal data
        plot_SIE_f1_score_degradation(sic_ds, model_version, patching_strategy_abbr, threshold=sie_threshold)
    end_time_f1_degradation = time.perf_counter()
    print(f"Elapsed time for F1-Score Degradation Plot: {end_time_f1_degradation - start_time_f1_degradation:.2f} seconds")

    # 8. Plot SIE Degradation (SIE as the area that is ice in km^2)
    # This function plots MAE and RMSE degradation over the forecast horizon for SIE in km^2.
    logging.info("\nDegradation of SIE (as square kilometers) over the forecast window ...")
    start_time_sie_kilometers_degradation = time.perf_counter()
    if not sie_ds.empty:
        plot_SIE_Kilometers_degradation(sie_ds, model_version, patching_strategy_abbr)
    end_time_sie_kilometers_degradation = time.perf_counter()
    print(f"Elapsed time for SIE Kilometers Degradation Plot: {end_time_sie_kilometers_degradation - start_time_sie_kilometers_degradation:.2f} seconds")

if PLOT_DAY_BY_DAY_METRICS:
    
    logging.info("\nPlotting per-day forecast analysis...")
    # --- Optional: Per-Day Analysis for Classification Metrics and Distributions ---
    # This section iterates through each forecast day to provide detailed metrics and plots.
    print("\n############################################")
    print("\n#   PER-DAY FORECAST ANALYSIS (Optional)   #")
    print("\n############################################")
    start_time_per_day_analysis = time.perf_counter()
    for day in range(1, FORECAST_HORIZON + 1): # Loop through each forecast day
        df_day = sic_ds[sic_ds['forecast_step'] == day]
        if not df_day.empty:
            print(f"\n--- Metrics for Forecast Day {day} ---")
            
            # Log Classification Report for specific day
            log_classification_report(df_day['actual'].values, df_day['predicted'].values, threshold=sie_threshold)
            
            # Plot SIC distribution for specific day
            plot_actual_vs_predicted_sic_distribution(
                df_day['actual'].values, df_day['predicted'].values, model_version, patching_strategy_abbr, num_bins=50, title_suffix=f" (Day {day})"
            )

            # Plot Confusion Matrix for specific day
            plot_sie_confusion_matrix(sic_ds, sie_threshold, model_version, patching_strategy_abbr, forecast_day=day)
            
            # Plot ROC Curve for specific day
            plot_roc_curve(sic_ds, model_version, patching_strategy_abbr, sie_threshold, forecast_day=day)
            
            end_time_per_day_analysis = time.perf_counter()
        print(f"Elapsed time for Per-Day Forecast Analysis: {end_time_per_day_analysis - start_time_per_day_analysis:.2f} seconds")

    # END OF EVALUATION
    end_full_evaluation = time.perf_counter()
    print(f"Elapsed time for FULL EVALUATION: {end_full_evaluation - start_full_evaluation:.2f} seconds")
    
    print("\nEvaluation complete.")

    # Restore stdout
    sys.stdout = sys.__stdout__
    
    # Now, write the captured output to the file
        f.write(captured_output.getvalue())
    
    print(f"Metrics saved as {model_version}_Metrics.txt")