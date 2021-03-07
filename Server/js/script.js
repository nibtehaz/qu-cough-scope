const record = document.querySelector('#record');

const soundClips = document.querySelector('.sound-clips');
const canvas = document.querySelector('.visualizer');
const mainSection = document.querySelector('.main-controls');

// disable stop button while not recording




if (navigator.mediaDevices.getUserMedia) {
    console.log('getUserMedia supported.');
  
    const constraints = { audio: true };
    let chunks = [];
  
    let onSuccess = function(stream) {
      const mediaRecorder = new MediaRecorder(stream);
  
      //visualize(stream);
  
      record.onclick = function() {
        mediaRecorder.start();
        console.log(mediaRecorder.state);
        console.log("recorder started");
        record.style.background = "red";
  
        
        record.disabled = true;
      }
  
      /*stopp.onclick = function() {
        mediaRecorder.stop();
        console.log(mediaRecorder.state);
        console.log("recorder stopped");
        record.style.background = "";
        record.style.color = "";
        // mediaRecorder.requestData();
  
        
        record.disabled = false;
      }*/
  
      mediaRecorder.onstop = function(e) {
        console.log("data available after MediaRecorder.stop() called.");
  

        ///audio.controls = true;
        const blob = new Blob(chunks, { 'type' : 'audio/wav' });
        console.log(chunks);
        chunks = [];

        const audioURL = window.URL.createObjectURL(blob);
        ///audio.src = audioURL;
        console.log(blob);
        alert(blob['size']);
        console.log("recorder stopped");
  


      }
  
      mediaRecorder.ondataavailable = function(e) {
        chunks.push(e.data);
      }
    }
  
    let onError = function(err) {
      alert('The following error occured: ' + err);
    }
  
    navigator.mediaDevices.getUserMedia(constraints).then(onSuccess, onError);
  
  } else {
     alert('getUserMedia not supported on your browser!');
  }



  ////////////

  fetch(`https://example.com/upload.php`, {method:"POST", body:blobData})
            .then(response => {
                if (response.ok) return response;
                else throw Error(`Server returned ${response.status}: ${response.statusText}`)
            })
            .then(response => console.log(response.text()))
            .catch(err => {
                alert(err);
            });