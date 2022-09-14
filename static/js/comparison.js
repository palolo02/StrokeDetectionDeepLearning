models = []
images = []
current_model = 0
current_image = 0

d3.json("/api/v1/models").then((incomingData) =>{
/* Load the available patients on the server and initialize the slider on the main page  */
    models = incomingData.models;
    console.log(models)
    document.getElementById("current_model").innerHTML = models[current_model]

        
    d3.json("/api/v1/models/images"+"?model="+models[current_model]).then((incomingImages) =>{
        /* Load the available patients on the server and initialize the slider on the main page  */
            images = incomingImages.images;
            no_images = images.length;
            console.log(models)
            console.log(no_images)
            _refreshImages()
        });
});

function changeModel(){
    /* Select another model from the data available on the server */
    current_model += 1
    if (current_model > models.length-1)    
        current_model = 0
    
    document.getElementById("current_model").innerHTML = models[current_model]
    _refreshImages()
}

function _refreshImages(){
    
    var input = document.getElementById("input")
    var mask = document.getElementById("mask")
    var prediction = document.getElementById("prediction")
    var comparison = document.getElementById("comparison")
    
    input.src=`static/data/validation/${models[current_model]}/input/${images[current_image]}`;
    mask.src=`static/data/validation/${models[current_model]}/mask/${images[current_image]}`;    
    prediction.src=`static/data/validation/${models[current_model]}/prediction/${images[current_image]}`;
    comparison.src=`static/data/validation/${models[current_model]}/comparison/${images[current_image]}`;
    
    document.getElementById("image_id").innerHTML = images[current_image]
    
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
}

function changeImage(){
    /* Select another model from the data available on the server */
    current_image += 1
    if (current_image > images.length-1)    
    current_image = 0
    _refreshImages()
}