$(document).ready(function () {
    // Set up tooltips
    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
    });

    // Swap up and down arrows on button click
    // TODO: functionality for every molecule that is shown
    let buttonNames = ["background", "use", "local", "drugbank"];
    buttonNames.forEach(function (buttonName) {
            $(`#${buttonName}CollapseButton`).click(function () {
                let arrow_div = $(`#${buttonName}Arrow`);
                arrow_div.toggleClass("arrow-up");
                arrow_div.toggleClass("arrow-down");
            });
        }
    );

    // Selection of SMILES input type
    $("#textButton").click(function () {
        $("#textInputForm").show();
        $("#textSmilesInput").prop('required', true);
        $("#fileInputForm").hide();
        $("#fileSmilesInput").prop('required', false);
        $("#drawInputForm").hide();
        $("#drawSmilesInput").prop('required', false);
        $("#drawSmilesInput").val('');
    });
    $("#fileButton").click(function () {
        $("#textInputForm").hide();
        $("#textSmilesInput").prop('required', false);
        $("#textSmilesInput").val('');
        $("#fileInputForm").show();
        $("#fileSmilesInput").prop('required', true);
        $("#drawInputForm").hide();
        $("#drawSmilesInput").prop('required', false);
        $("#drawSmilesInput").val('');
    });
    $("#drawButton").click(function () {
        $("#textInputForm").hide();
        $("#textSmilesInput").prop('required', false);
        $("#textSmilesInput").val('');
        $("#fileInputForm").hide();
        $("#fileSmilesInput").prop('required', false);
        $("#drawInputForm").show();
        $("#drawSmilesInput").prop('required', true);
    });
    $("#exampleButton").click(function () {
        $("#textInputForm").show();
        $("#textSmilesInput").prop('required', true);
        $("#textSmilesInput").val("O(c1ccc(cc1)CCOC)CC(O)CNC(C)C");
        $("#fileInputForm").hide();
        $("#fileSmilesInput").prop('required', false);
        $("#drawInputForm").hide();
        $("#drawSmilesInput").prop('required', false);
        $("#drawSmilesInput").val('');
    });

    // Convert molecule drawing to SMILES
    $("#convertToSmiles").click(function () {
        $("#drawSmilesInput").val(jsmeApplet.smiles());
    });

    // ATC selection search
    $("#atc-selection").on("keyup", function () {
        var value = $(this).val().toLowerCase();
        $(".atc-item").filter(function () {
            $(this).toggle($(this).text().toLowerCase().indexOf(value) > -1)
        });
    });

    // ATC selection click
    function clickATC(atc_code) {
        $.ajax({
            url: "/drugbank_plot?atc_code=" + atc_code,
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

    // ATC selection click
    $('.atc-item').click(function () {
        var atc_code = $(this).text();
        clickATC(atc_code);
    });

    // DrugBank selection click
    function clickDrugBank(task, axis) {
        $.ajax({
            url: `/drugbank_plot?${axis}_task=` + task,
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
});

