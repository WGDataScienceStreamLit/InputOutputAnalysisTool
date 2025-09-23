import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from iot_leontief_python import *
from scripts.app_funcs import *

# Set the page setting to wide
st.set_page_config(layout="wide", page_title="IO Analysis Tool", page_icon="📈")

# Intro text
st.write(display_text("./assets/text/intro.txt"))

# Tab layout for the rest of the app
guide, loading_page, effects = st.tabs(["User Guide", "IOT", "Scenario Planning"])

# Guide section
with guide:
    # Write user guide on the left and caveats on the right
    left_g, right_g = st.columns([1, 1])
    with left_g:
        st.write(display_text("./assets/text/user_guide.txt"))
    with right_g:
        st.write(display_text("./assets/text/caveats.txt"))
        acknowledge = st.checkbox("I accept")

# Only run rest of application if acknowledged
if acknowledge:
    # Tab to load in io matrix
    with loading_page:
        st.write(display_text("./assets/text/loading.txt"))

        # File uploader for user df
        new_matrix = st.file_uploader(
            "Upload a new matrix to use in the analysis", type="csv"
        )

        # Otherwise use dummy file
        if new_matrix is None:
            df = pd.read_csv("./data/input/iot_tool_placeholder_table.csv")

            # Dummy column names as defaults
            names_col = "Industry Purchases ↓  Industry Sales →"
            demand_col = "Total intermediate use"
            output_row = "Total output at basic prices"
            hh_use_col = "Households final consumption expenditure"
            disp_inc_row = "Compensation of employees"
            gdhi_input = 0

        else:
            df = pd.read_csv(new_matrix)  # read the user df

            # Prompt user for the column names needed
            names_col = st.selectbox(
                "Please select the name of the column containing the sector names/codes: ",
                options=df.columns,
            )
            demand_col = st.selectbox(
                "Please select a column to use as the Total Intermediate Use: ",
                options=df.columns,
            )
            output_row = st.selectbox(
                "Please select a column to use as the Total Use: ",
                options=["", *df[names_col]],
            )

            # Pause the application while user completes selection
            if names_col == demand_col or names_col == output_row:
                st.stop()

            # Allow for optional columns that the user may add if doing T2 modelling
            st.subheader("(Optional) Induced income locations")
            hh_use_col = st.selectbox(
                "Please select the column for household demand (household final consumption expenditure): ",
                options=["", *df.columns],
            )
            hh_use_col = None if hh_use_col == "" else hh_use_col

            disp_inc_row = st.selectbox(
                "Please select the row for labour input (income): ",
                options=["", *df[names_col]],
            )
            disp_inc_row = None if disp_inc_row == "" else disp_inc_row

            # Optional GDHI constant
            gdhi_input = st.number_input(
                "Optional: Enter a constant to use as a denominator when calculating the household expenditure coefficients for the Type 2 multiplier calculation. See the User Guide for further information."
            )

        # Only allow induced modelling if induced fields are available
        show_induced_option = not ((hh_use_col is None) or (disp_inc_row is None))

        # Restore old df on click
        refresh_df = st.button("Restore default data frame")

        if refresh_df:
            df = pd.read_csv("./data/input/iot_tool_placeholder_table.csv")

        # Display the dataframe that is in use
        display_df = create_display_df(df, names_col)
        st.dataframe(display_df)

        # Option to download
        st.download_button(
            "Download the current IOT to use as a template",
            convert_df(display_df),
            "default_matrix.csv",
            "text/csv",
        )

        # Information about the template IO table
        st.write(display_text("./assets/text/io_template.txt"))
        im = Image.open("assets/IOTemplate.png")
        im = im.resize((600, 400))
        st.image(im)

    # Set up parts for IOT Model
    output, sectors, disposable_income, hh_demand, io_matrix = get_components(
        df, demand_col, output_row, names_col, hh_use_col, disp_inc_row
    )

    # If no GDHI use sum of HH as default
    if gdhi_input == 0:
        gdhi = np.sum(hh_demand)
    else:
        gdhi = gdhi_input

    # Modelling tab
    with effects:
        st.write(display_text("./assets/text/sectors.txt"))

        # Give option for whether or not to induced effects
        if show_induced_option:
            st.write("### Activate Type 2 Modelling")
            incind = st.checkbox("Include induced effects (Use Type 2 Modelling)")
        else:
            incind = False

        # Option to view total multipliers
        view_mult = st.expander("View: Analysis Leontief Inverse multipliers.")

        with view_mult:
            if incind:
                # Calculate T2 multipliers
                mults, mults_mat = t2_multipliers(
                    io_matrix, output, hh_demand, disposable_income, gdhi
                )
            else:
                # Calculate T1 multipliers
                mults, mults_mat = t1_multipliers(io_matrix, output)

            # Display with a color map
            mults_df = pd.DataFrame(mults, columns=["Multipliers"], index=sectors)
            plt.rcParams["image.cmap"] = "Oranges"
            st.dataframe(mults_df.style.background_gradient(cmap=None))

        # Layout editable demand left, analysis right
        edit_demand, modelled_result = st.columns(
            [1, 3]
        )  # set up columns for each area

        with edit_demand:
            st.markdown("### Model a scenario")

            # Allow the user to edit the demand
            user_demand = st.data_editor(
                pd.DataFrame([0] * len(sectors), index=sectors, columns=["Demand"]),
                use_container_width=True,
                height=1980,
            )

        # Once the demand has been edited
        if (user_demand["Demand"] == 0).all():
            pass
        else:
            # Model the scenario
            modelled_demand = model_scenario(mults_mat, user_demand)

            # Isolate effects for visualisation
            # Direct Demand is the user input
            direct_demand = user_demand["Demand"]

            # Isolate the other demand types
            if incind:
                # If a T2 model
                # induced = t2modelled - t1modelled
                # indirect = modelled - direct - induced
                # Model T1 Outputs
                _, t1_mat = t1_multipliers(io_matrix, output)
                t1_modelled = model_scenario(t1_mat, user_demand)

                # Subtract the direct demand from them to find indirect demand
                indirect_demand = t1_modelled - direct_demand

                # Then subtract them from the model to get Induced Demand
                induced_demand = modelled_demand - t1_modelled

            else:
                # If a T1 model then modelled - direct = indirect
                induced_demand = None
                indirect_demand = modelled_demand - direct_demand

            # Flatten original output vec
            output_vis = output.to_numpy().flatten()

            print(mults_mat)

            # Construct into DF:
            final_output = pd.DataFrame(
                {
                    "OriginalOutput": output_vis[: len(sectors)],
                    "DirectDemand": direct_demand[: len(sectors)],
                    "IndirectDemand": indirect_demand[: len(sectors)],
                    "InducedEffect": induced_demand[: len(sectors)]
                    if induced_demand is not None
                    else [0] * len(sectors),
                },
                index=sectors,
            )

        # Right side displaying the results
        with modelled_result:
            if (user_demand["Demand"] == 0).all():
                st.warning(
                    "Please use the table on the left to begin modelling a new scenario."
                )
            else:
                # Tabbed layout for bar, filtered bar, table, download option
                modelling, compare, table, download = st.tabs(
                    [
                        "📈 View outputs from model",
                        "📊 Compare with current outputs",
                        "🔭 View as a table",
                        "👇 Download",
                    ]
                )
                with modelling:
                    # Provide option for viewing as proportion of current demand
                    view_pc = st.selectbox(
                        "View the scenario as a proportion of current output: ",
                        options=[
                            "View as actual output for the scenario",
                            "View as a proportion of the current output in the IOT",
                        ],
                    )

                    prop = (
                        view_pc
                        == "View as a proportion of the current output in the IOT"
                    )

                    # Build and style plots based on prop or actual
                    fig = model_bar(
                        100 * direct_demand / output_vis if prop else direct_demand,
                        100 * indirect_demand / output_vis if prop else indirect_demand,
                        100 * induced_demand / output_vis
                        if prop and incind
                        else induced_demand,
                        sectors.copy(),
                        modelled_demand,
                        incind,
                    )

                    fig = update_plots(fig, "Modelled Scenario Demand by Sector")
                    st.plotly_chart(fig, use_container_width=True)

                # For comparison with current output
                with compare:
                    # Get a filter list of categories to include
                    filters = st.multiselect(
                        "Sectors to include in the visualisation", options=sectors
                    )

                    # Build and style the plot
                    comp = comp_bar(
                        direct_demand,
                        indirect_demand,
                        induced_demand,
                        sectors.copy(),
                        output_vis,
                        incind,
                        filters,
                    )

                    comp = update_plots(
                        comp, "Current and Modelled Demand by Sector", height=1200
                    )
                    st.plotly_chart(comp, use_container_width=True)

                # To view as table for editing in Excel
                with table:
                    st.dataframe(final_output, use_container_width=True, height=1980)

                # For download of scenario
                with download:
                    st.download_button(
                        "Download the model results.",
                        convert_df(final_output),
                        "modelling_result.csv",
                        "text/csv",
                    )


else:
    # If acknowledgement not checked, write warning on every page
    for tab in [loading_page, effects]:
        with tab:
            st.error(
                "**Please acknowledge that you have read the User Guide and Caveats at the bottom of the User Guide tab.**"
            )

# Add a footer at the bottom of all tabs.
st.write(display_text("./assets/text/footer.txt"))
