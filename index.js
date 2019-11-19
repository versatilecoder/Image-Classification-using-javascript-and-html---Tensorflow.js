let net;
//document.getElementById("webcam").style.display="block";
//document.getElementById("cam").style.display="block";

const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

  var openFile = function(file) {

    var input = file.target;

    var reader = new FileReader();
    reader.onload = function(){
      var dataURL = reader.result;
      var output = document.getElementById('output');
      output.src = dataURL;
    };
    reader.readAsDataURL(input.files[0]);
	
	   var imgtext = document.getElementById('output').value;
	$("#img").attr("src",imgtext);
  };
  
  $('body').on('change', '#classname', function() {

	 alert($("#classname").val());
	$(".addname").attr("id", "class-"+$("#classname").val());

  });
  
async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Successfully loaded model');

  // Create an object from Tensorflow.js data API which could capture image 
  // from the web camera as Tensor.
   const webcam = await tf.data.webcam(webcamElement);
 while (true) {
    const img = await webcam.capture();
    const result = await net.classify(img);

    document.getElementById('console').innerText = `
      prediction: ${result[0].className}\n
      probability: ${result[0].probability}
    `;
    // Dispose the tensor to release the memory.
    img.dispose();

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }
  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = async classId => {
    // Capture an image from the web camera.
    const img = await webcam.capture();
const result1 = await net.classify(img);
 document.getElementById('console').innerText = `
      prediction: ${result1[0].className}\n
      probability: ${result1[0].probability}
    `;
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(img, 'conv_preds');

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);

    // Dispose the tensor to release the memory.
    img.dispose();
  };

  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample($("#classname").val()));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();
	
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(img, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['A', 'B', 'C'];
	  classes.push($("#classname").val());
	  
      document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `;

      // Dispose the tensor to release the memory.
      img.dispose();
    }

    await tf.nextFrame();
  }
}
async function appupload() {

  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Successfully loaded model');
  //var imgtext = document.getElementById('imgtext').value;
  //alert(imgtext);
//$("#img").attr("src",imgtext);
  // Make a prediction through the model on our image.
  const imgEl = document.getElementById('output');
  console.log(imgEl);
  const result = await net.classify(imgEl);
  console.log(result);
  $("#results").append("<table border='1'><tr><td>Name</td><td>Probability</td></tr>");
  $.each(result, function(key, value) {
    console.log("Value is "+key+" Value"+ value);
	$("#results").append("<tr border='1'><td>"+result[key].className+ "</td><td>"+result[key].probability+"</td></tr>");
});
 $("#results").append("</table>");
  /*Object.entries(result).forEach(
    ([className, probability]) =>  $("#results").text(result[className]+" Probability : "+result[probability])
	
);
 /*for(int i=0;i<result.size;i++){
 $("#results").text(result[i].className+" Probability : "+result[i].probability);
  }*/
  
}



app();