import streamlit as st
import pandas as pd
import glob
import os
from pathlib import Path

st.set_page_config(page_title="Results Viewer", layout="wide")

def parse_results_file(file_path):
    """Parse results CSV file and extract metadata from header comments."""
    metadata = {}

    # Read the file to extract metadata from comments
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                # Parse metadata from comment lines
                if ':' in line:
                    parts = line[1:].strip().split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        metadata[key] = value
            else:
                # Stop when we reach the actual CSV data
                break

    # Read the actual CSV data (skip comment lines)
    df = pd.read_csv(file_path, comment='#')

    return df, metadata

def main():
    st.title("🔍 Results Viewer")

    # Find all result files in the directory
    result_files = glob.glob("results_*.csv")

    if not result_files:
        st.error("No result files found in the directory. Looking for files matching 'results_*.csv'")
        return

    # Sort files by modification time (newest first)
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Create a dropdown to select the file
    st.sidebar.header("Select Results File")
    selected_file = st.sidebar.selectbox(
        "Choose a results file:",
        result_files,
        format_func=lambda x: f"{Path(x).name} ({pd.Timestamp.fromtimestamp(os.path.getmtime(x)).strftime('%Y-%m-%d %H:%M:%S')})"
    )

    # Load the selected file
    try:
        df, metadata = parse_results_file(selected_file)

        # Display metadata in the sidebar
        st.sidebar.header("📊 Metadata")
        for key, value in metadata.items():
            st.sidebar.metric(key, value)

        # Main content area
        st.header(f"Results from: {Path(selected_file).name}")

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            correct_count = df['correct'].sum() if 'correct' in df.columns else 0
            st.metric("Correct Predictions", correct_count)
        with col3:
            incorrect_count = len(df) - correct_count
            st.metric("Incorrect Predictions", incorrect_count)
        with col4:
            accuracy = correct_count / len(df) if len(df) > 0 else 0
            st.metric("Accuracy", f"{accuracy:.2%}")

        # Filters
        st.subheader("🔎 Filters")
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            show_correct = st.checkbox("Show Correct", value=True)
            show_incorrect = st.checkbox("Show Incorrect", value=True)

        with filter_col2:
            if 'predicted_label' in df.columns:
                predicted_filter = st.multiselect(
                    "Predicted Label",
                    options=sorted(df['predicted_label'].unique()),
                    default=sorted(df['predicted_label'].unique())
                )
            else:
                predicted_filter = None

        with filter_col3:
            if 'ground_truth' in df.columns:
                ground_truth_filter = st.multiselect(
                    "Ground Truth",
                    options=sorted(df['ground_truth'].unique()),
                    default=sorted(df['ground_truth'].unique())
                )
            else:
                ground_truth_filter = None

        # Apply filters
        filtered_df = df.copy()
        if 'correct' in df.columns:
            if not show_correct:
                filtered_df = filtered_df[filtered_df['correct'] == False]
            if not show_incorrect:
                filtered_df = filtered_df[filtered_df['correct'] == True]

        if predicted_filter is not None and 'predicted_label' in df.columns:
            filtered_df = filtered_df[filtered_df['predicted_label'].isin(predicted_filter)]

        if ground_truth_filter is not None and 'ground_truth' in df.columns:
            filtered_df = filtered_df[filtered_df['ground_truth'].isin(ground_truth_filter)]

        # Search functionality
        search_query = st.text_input("🔍 Search in messages:", "")
        if search_query:
            filtered_df = filtered_df[filtered_df['message'].str.contains(search_query, case=False, na=False)]

        st.subheader(f"Displaying {len(filtered_df)} of {len(df)} samples")

        # Display mode selection
        display_mode = st.radio(
            "Display Mode:",
            ["Detailed Cards", "Table View"],
            horizontal=True
        )

        if display_mode == "Detailed Cards":
            # Display each datapoint as a detailed card
            for idx, row in filtered_df.iterrows():
                with st.expander(f"Sample #{idx + 1} - {'✅ Correct' if row.get('correct', False) else '❌ Incorrect'}"):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown("**Message:**")
                        st.write(row['message'])

                    with col2:
                        if 'predicted_label' in row:
                            label_color = "🟢" if row.get('predicted_label') == 1 else "🔵"
                            st.markdown(f"**Predicted:** {label_color} {row['predicted_label']}")

                        if 'ground_truth' in row:
                            truth_color = "🟢" if row.get('ground_truth') == 1 else "🔵"
                            st.markdown(f"**Ground Truth:** {truth_color} {row['ground_truth']}")

                        if 'correct' in row:
                            st.markdown(f"**Correct:** {'✅' if row['correct'] else '❌'}")

                    # Display both original and new reasoning if available
                    if 'original_reasoning' in row and pd.notna(row['original_reasoning']):
                        st.markdown("**Original Reasoning (Sonnet 4.5 - Initial):**")
                        st.info(row['original_reasoning'])

                    if 'new_reasoning' in row and pd.notna(row['new_reasoning']):
                        st.markdown("**New Reasoning:**")
                        st.success(row['new_reasoning'])

                    # Fallback for older format with just 'reasoning'
                    if 'reasoning' in row and pd.notna(row['reasoning']) and 'new_reasoning' not in row:
                        st.markdown("**Reasoning:**")
                        st.info(row['reasoning'])

        else:  # Table View
            # Display as a table with formatting
            display_df = filtered_df.copy()

            # Truncate long messages for table view
            if 'message' in display_df.columns:
                display_df['message'] = display_df['message'].apply(
                    lambda x: x[:100] + '...' if len(str(x)) > 100 else x
                )

            # Truncate reasoning columns
            if 'original_reasoning' in display_df.columns:
                display_df['original_reasoning'] = display_df['original_reasoning'].apply(
                    lambda x: x[:100] + '...' if pd.notna(x) and len(str(x)) > 100 else x
                )

            if 'new_reasoning' in display_df.columns:
                display_df['new_reasoning'] = display_df['new_reasoning'].apply(
                    lambda x: x[:100] + '...' if pd.notna(x) and len(str(x)) > 100 else x
                )

            # Fallback for older format
            if 'reasoning' in display_df.columns:
                display_df['reasoning'] = display_df['reasoning'].apply(
                    lambda x: x[:100] + '...' if pd.notna(x) and len(str(x)) > 100 else x
                )

            st.dataframe(
                display_df,
                use_container_width=True,
                height=600
            )

        # Download filtered results
        st.subheader("💾 Export")
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Results as CSV",
            data=csv,
            file_name=f"filtered_{Path(selected_file).name}",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
