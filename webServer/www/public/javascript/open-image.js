$(document).ready(function(){

  // dislplays the clicked on image in a modal
	$('.pop').on('click', function() {
		console.log($(this).find('img').attr('src'));
		$('#imagepreview').attr('src', $(this).find('img').attr('src'));
		$('#exploremodal').modal('show');
	});

  // calls readURL function when user uploads a photo
  $("#uploadPhoto").change(function () {
      readURL(this, 'uploaded-image');
  });

  // calls readURL function when user uploads a video
  $("#uploadVideo").change(function () {
      readURL(this, 'uploaded-image');
  });

  // displays the uploaded image or video in a modal
  function readURL(input, id) {
      if (input.files && input.files[0]) {
          var reader = new FileReader();

          reader.onload = function (e) {
              $('#' + id).attr('src', e.target.result);
              $('#homeModal').modal('show');
          }

          reader.readAsDataURL(input.files[0]);
      }
  }
})


