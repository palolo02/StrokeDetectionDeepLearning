index = 0
max_img = 256
intervalId = null
patients = []
current_patient = 0

d3.json("/api/v1/patients").then((incomingData) =>{
/* Load the available patients on the server and initialize the slider on the main page  */
    patients = incomingData.patients;
    console.log(patients)

    var handle = $( "#custom-handle" );
    $( "#slider" ).slider({
        min: 0,
        max: patients[current_patient]["files"],
        create: function() {
            handle.text( $( this ).slider( "value" ) );
        },
        slide: function( event, ui ) {
            handle.text( ui.value );
            index = ui.value
        },
        change: function( event, ui ) {
            handle.text( ui.value );
            index = ui.value
            document.getElementById("image_id").innerHTML = index
            changeImages()
        }
    });
    changeImages()
});


function changePatient(){
/* Select another patient from the data available on the server */
    current_patient += 1
    if (current_patient > patients.length-1)    
        current_patient = 0
    document.getElementById("current_patient").innerHTML = patients[current_patient]["name"]
    max_img = patients[current_patient]["files"]
    document.getElementById("no_slices").innerHTML = max_img
    // Update slider
    $( "#slider" ).slider( "option", "max", max_img)
    //$( ".selector" ).slider( "option", "max", max_img);
    changeImages()
}


function changeImages()
/* Update the images layout according to the selected patient */
{
    var input = document.getElementById("input")
    var mask = document.getElementById("mask")
    var prediction = document.getElementById("prediction")
    var comparison = document.getElementById("comparison")

    index_img = String(index).padStart(4, '0'); // '0010'
    input.src=`static/data/input/${patients[current_patient]["name"]}/${index_img}.png`;
    if (patients[current_patient]["mask"])
        mask.src=`static/data/mask/${patients[current_patient]["name"]}/${index_img}.png`;
    else
        mask.src=`static/data/folder/0255.png`;
    prediction.src=`static/data/prediction/${patients[current_patient]["name"]}/${index_img}.png`;
    comparison.src=`static/data/comparison/${patients[current_patient]["name"]}/${index_img}.png`;
    
    // Measure the dsc between the prediction and the mask
    if (patients[current_patient]["mask"])
        $.ajax({ 
            type: "GET", 
            url: "/api/v1/images/dsc",
        
            // parameter here
            data : {img1: mask.src, img2: prediction.src},
            success: function(json){
                console.log(json["dsc"]);
                document.getElementById("dsc_id").innerHTML = json["dsc"]     
            } ,
            error:function (request, err, ex) {
                console.log("There was an error calling the route")
            }
        });
    else
        document.getElementById("dsc_id").innerHTML = "Ground-truth not available"
    return false;
}


function changeAutomaticImages()
/* Enable automatic change on the images layout according to the selected patient */
{
    intervalId = window.setInterval(function(){
        /// call your function here
        index += 1
        if(index == max_img) 
            index=0
        document.getElementById("image_id").innerHTML = index
        $( "#custom-handle" ).text(index);
        $( "#slider" ).slider( "option", "value", index)
        changeImages()
        
        
      }, 500);
}


function stop(){
/* Disable automatic change on the images layout */
    clearInterval(intervalId)
}