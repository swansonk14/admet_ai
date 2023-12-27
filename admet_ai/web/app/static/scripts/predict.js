$(document).ready(function () {
    // Set up tooltips
    $(function () {
        $('[data-toggle="tooltip"]').tooltip({
            container: "body"
        });
    });

    // Swap up and down arrows on button click
    let buttonNames = ["background", "use", "local", "drugbank"];
    buttonNames.forEach(function (buttonName) {
            $(`#${buttonName}-collapse-button`).click(function () {
                let arrow_div = $(`#${buttonName}-arrow`);
                arrow_div.toggleClass("arrow-up");
                arrow_div.toggleClass("arrow-down");
            });
        }
    );

    // Resize JSME applet
    function resizeJSME() {
        // Get the new container dimensions
        var container = document.getElementById("jsme_container");
        var width = container.offsetWidth;
        var height = container.offsetHeight;

        // Update JSME applet size
        jsmeApplet.setSize(width + "px", height + "px");
    }

    // Resize JSME applet on window size change
    window.addEventListener("resize", function () {
        resizeJSME();
    });

    // Selection of SMILES input type
    $("#text-button").click(function () {
        $("#text-input-form").show();
        $("#text-smiles-input").prop('required', true);
        $("#file-input-form").hide();
        $("#file-smiles-input").prop('required', false);
        $("#draw-input-form").hide();
    });
    $("#file-button").click(function () {
        $("#text-input-form").hide();
        $("#text-smiles-input").prop('required', false);
        $("#text-smiles-input").val('');
        $("#file-input-form").show();
        $("#file-smiles-input").prop('required', true);
        $("#draw-input-form").hide();
    });
    $("#draw-button").click(function () {
        $("#text-input-form").hide();
        $("#text-smiles-input").prop('required', false);
        $("#text-smiles-input").val('');
        $("#file-input-form").hide();
        $("#file-smiles-input").prop('required', false);
        $("#draw-input-form").show();
        resizeJSME();
    });
    $("#example-button").click(function () {
        $("#text-input-form").show();
        $("#text-smiles-input").prop('required', true);
        $("#text-smiles-input").val("O(c1ccc(cc1)CCOC)CC(O)CNC(C)C");
        $("#file-input-form").hide();
        $("#file-smiles-input").prop('required', false);
        $("#draw-input-form").hide();
    });

    // Convert molecule drawing to SMILES
    $("#predict-button").click(function () {
        $("#draw-smiles-input").val(jsmeApplet.smiles());
    });

    // ATC selection search
    $("#atc-selection").on("keyup", function () {
        let value = $(this).val().toLowerCase();
        $(".atc-item").filter(function () {
            $(this).toggle($(this).text().toLowerCase().indexOf(value) > -1)
        });
    });

    // ATC selection click
    function clickATC(atc_code) {
        $.ajax({
            url: "/set_atc_code?atc_code=" + atc_code,
            type: "POST",
            dataType: "json",
            success: function (response) {
                if (response.drugbank_size_string) {
                    document.getElementById("selected-atc-code").innerHTML = atc_code;
                    document.getElementById("drugbank-size").innerHTML = response.drugbank_size_string;
                }
            },
            error: function (error) {
                console.error("Failed to fetch ATC code or DrugBank size:", error);
            }
        });
    }

    // ATC selection click
    $('.atc-item').click(function () {
        let atc_code = $(this).text();
        clickATC(atc_code);
    });

    // DrugBank selection click
    function clickDrugBank(task, axis) {
        let atc_code = $("#applied-atc-code").text();
        $.ajax({
            url: `/drugbank_plot?${axis}_task=${task}&atc_code=${atc_code}`,
            type: "GET",
            dataType: "json",
            success: function (response) {
                if (response.svg) {
                    document.getElementById("drugbank-plot-inner").innerHTML = response.svg;
                }
            },
            error: function (error) {
                console.error("Failed to fetch DrugBank SVG:", error);
            }
        });
    }

    // DrugBank selection search and click on both axes
    let axes = ["x", "y"];
    axes.forEach(function (axis) {
        $(`#drugbank-${axis}-axis-selection`).on("keyup", function () {
            let value = $(this).val().toLowerCase();
            $(`.drugbank-${axis}-axis-item`).filter(function () {
                $(this).toggle($(this).text().toLowerCase().indexOf(value) > -1)
            });
        });

        $(`.drugbank-${axis}-axis-item`).click(function () {
            let task = $(this).text();
            clickDrugBank(task, axis);
        });
    });

    $("#molecule-form").on("submit", function () {
        $("#spinner-overlay").css("visibility", "visible");
    });
});

