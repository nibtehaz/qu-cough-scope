var mediaRecorder;
// disable stop button while not recording

var recording_cough = false;
var recording_breath = false;


function init_media_recorder() {

    if (navigator.mediaDevices.getUserMedia) {
        console.log('getUserMedia supported.');

        const constraints = { audio: true };
        let chunks = [];

        let onSuccess = function (stream) {
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.onstop = function (e) {
                console.log("data available after MediaRecorder.stop() called.");


                ///audio.controls = true;
                //const blob = new Blob(chunks, { 'type': 'audio/ogg; codecs=opus' });
                var blob = new Blob(chunks, { 'type': 'audio/wav' });
                console.log(chunks);
                chunks = [];
                

                if (current_audio_index==0){
                    var audio_object = document.getElementById("cough_audio");
                    audio_object.src = window.URL.createObjectURL(blob);
                    recording_cough = false;                    
                    document.getElementById('cough_audio_div').style.visibility = "visible";
                }

                if (current_audio_index==1){
                    var audio_object = document.getElementById("breath_audio");
                    audio_object.src = window.URL.createObjectURL(blob);
                    recording_cough = false;                    
                    document.getElementById('breath_audio_div').style.visibility = "visible";
                }
                
                audioblobs[current_audio_index] = (blob);

                current_audio_index += 1;

                if (current_audio_index==2){
                    //submit();
                }

            }

            mediaRecorder.ondataavailable = function (e) {
                chunks.push(e.data);
            }
        }

        let onError = function (err) {
            alert('The following error occured: ' + err);
        }

        navigator.mediaDevices.getUserMedia(constraints).then(onSuccess, onError);

    } else {
        alert('getUserMedia not supported on your browser!');
    }
}

function start_record_cough() {
    

    console.log('recording_cough');
    console.log(recording_cough);
    if (recording_cough) {
        return;
    }

    mediaRecorder.start();
    console.log(mediaRecorder.state);
    console.log("recorder started");

    current_audio_index = 0;
    document.getElementById(`progress_cough`).style.width = `0%`;

    recording_cough = true;

    setTimeout(() => {
        increase_progressbar(100,'cough');
    }, 100);

    setTimeout(() => {
        mediaRecorder.stop();
    }, 10000);

    console.log('here');

}



function start_record_breath() {
    console.log('breath');
    console.log(recording_breath);
    if (recording_breath) {
        return;
    }

    mediaRecorder.start();
    console.log(mediaRecorder.state);
    console.log("recorder started");

    current_audio_index = 1;
    document.getElementById(`progress_breath`).style.width = `0%`;

    recording_cough = true;

    setTimeout(() => {
        increase_progressbar(150,'breath');
        console.log('call');
    }, 150);

    setTimeout(() => {
        mediaRecorder.stop();
    }, 15000);

    console.log('here');


}



function increase_progressbar(delay, obj_typ){

    if (document.getElementById(`progress_${obj_typ}_wrapper`).style.visibility==='hidden'){
        document.getElementById(`progress_${obj_typ}_wrapper`).style.visibility='visible';
    }
    
    var prcnt = parseInt(document.getElementById(`progress_${obj_typ}`).style.width);

    if(prcnt!=100){
        document.getElementById(`progress_${obj_typ}`).style.width = `${prcnt+1}%`;

        setTimeout( () => {increase_progressbar(delay, obj_typ)}, delay );
    }

    else{
        document.getElementById(`${obj_typ}_next_btn`).style.visibility='visible';

        
    }
    

}


  ////////////

/*fetch(`https://example.com/upload.php`, {method:"POST", body:blobData})
          .then(response => {
              if (response.ok) return response;
              else throw Error(`Server returned ${response.status}: ${response.statusText}`)
          })
          .then(response => console.log(response.text()))
          .catch(err => {
              alert(err);
          });*/