var userdata = {}
var symptomdata = {}
var audioblobs = [0, 0];
var current_audio_index;

document.addEventListener('DOMContentLoaded', function () {
    var elems = document.querySelectorAll('.modal');
    var instances = M.Modal.init(elems, {});
});







function init() {


}

function start_app() {

    var userid = getCookie('userid');

    current_audio_index = 0;

    document.getElementById('home').style.display = 'none';



    if (userid.length != 0) {
        userdata['userid'] = userid;
        document.getElementById('page5').style.display = 'block';
    }

    else {
        document.getElementById('page1').style.display = 'block';
    }




}

function next_page(cur_pg) {

    if (cur_pg == 1) {
        var radios = document.getElementsByName('sex_inp');

        userdata['sex'] = '';

        for (var i = 0, length = radios.length; i < length; i++) {
            if (radios[i].checked) {
                userdata['sex'] = radios[i].value;
            }
        }

        if (userdata['sex'].length == 0) {
            M.toast({ html: 'Please select an option' });
            return;
        }

    }

    if (cur_pg == 2) {
        var radios = document.getElementsByName('age_inp');

        userdata['age'] = '';

        for (var i = 0, length = radios.length; i < length; i++) {
            if (radios[i].checked) {
                userdata['age'] = radios[i].value;
            }
        }

        if (userdata['age'].length == 0) {
            M.toast({ html: 'Please select an option' });
            return;
        }
    }

    if (cur_pg == 3) {
        var radios = document.getElementsByName('disease_inp');

        userdata['disease'] = [];

        for (var i = 0, length = radios.length; i < length; i++) {
            if (radios[i].checked) {
                userdata['disease'].push(radios[i].value);
            }
        }

        if (userdata['disease'].length == 0) {
            M.toast({ html: 'Please select an option' });
            return;
        }
    }

    if (cur_pg == 4) {
        var radios = document.getElementsByName('smoke_inp');

        userdata['smoke'] = '';

        for (var i = 0, length = radios.length; i < length; i++) {
            if (radios[i].checked) {
                userdata['smoke'] = radios[i].value;
            }
        }

        if (userdata['smoke'].length == 0) {
            M.toast({ html: 'Please select an option' });
            return;
        }
    }

    if (cur_pg == 5) {
        var radios = document.getElementsByName('symp_inp');

        symptomdata['symptoms'] = [];

        for (var i = 0, length = radios.length; i < length; i++) {
            if (radios[i].checked) {
                symptomdata['symptoms'].push(radios[i].value);
            }
        }

        if (symptomdata['symptoms'].length == 0) {
            M.toast({ html: 'Please select an option' });
            return;
        }

    }

    if (cur_pg == 6) {
        var radios = document.getElementsByName('covidtest_inp');

        symptomdata['covidtest'] = '';

        for (var i = 0, length = radios.length; i < length; i++) {
            if (radios[i].checked) {
                symptomdata['covidtest'] = radios[i].value;
            }
        }

        if (symptomdata['covidtest'].length == 0) {
            M.toast({ html: 'Please select an option' });
            return;
        }
    }

    if (cur_pg == 7) {
        var radios = document.getElementsByName('covidinfct_inp');

        symptomdata['infection'] = '';

        for (var i = 0, length = radios.length; i < length; i++) {
            if (radios[i].checked) {
                symptomdata['infection'] = radios[i].value;
            }
        }

        if (symptomdata['infection'].length == 0) {
            M.toast({ html: 'Please select an option' });
            return;
        }

    }

    if (cur_pg == 8) {
        var radios = document.getElementsByName('hospital_inp');

        symptomdata['hospital'] = '';

        for (var i = 0, length = radios.length; i < length; i++) {
            if (radios[i].checked) {
                symptomdata['hospital'] = radios[i].value;
            }
        }

        if (symptomdata['hospital'].length == 0) {
            M.toast({ html: 'Please select an option' });
            return;
        }
    }


    if (cur_pg == 9) {
    
        submit();
    }


    clear_toasts();

    document.getElementById(`page${cur_pg}`).style.display = 'none';
    document.getElementById(`page${cur_pg + 1}`).style.display = 'block';



}


function init_tooltips() {
    var elems = document.querySelectorAll('.tooltipped');
    var instances = M.Tooltip.init(elems, {});
}


function clear_toasts() {

    M.Toast.dismissAll();

}


function set_cookie(token) {
    document.cookie = `userid=${token}; expires=Sat, 18 Dec 3013 12:00:00 UTC;`
}



function getCookie(cname) {
    var name = cname + "=";
    var decodedCookie = decodeURIComponent(document.cookie);
    var ca = decodedCookie.split(';');
    for (var i = 0; i < ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') {
            c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
            return c.substring(name.length, c.length);
        }
    }
    return "";
}


function submit() {

    var fd = new FormData(document.getElementById("audio_upload"));

    fd.append('userdata', JSON.stringify(userdata));
    fd.append('symptomdata', JSON.stringify(symptomdata));
    
    $.ajax({
        url: 'audio_upload',
        type: "POST",
        processData: false,
        contentType: false,
        data: fd,
        success: function (data) {
            console.log(data);
        },
        error: function (jXHR, textStatus, errorThrown) {
            alert(errorThrown);

        }
    }).done(function (data) {
        console.log(data);
        set_cookie(data['userid']);
        if (data['predicted_class'] === 'covid') {

            window.location.href = "/positive";
        }
        else {
            window.location.href = "/negative";
        }

    });


}