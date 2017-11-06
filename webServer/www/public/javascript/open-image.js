var contentURL;

$(document).ready(function(){

  // dislplays the clicked on image in a modal
	$('.pop').on('click', function() {
		console.log($(this).find('img').attr('src'));
		$('#imagepreview').attr('src', $(this).find('img').attr('src'));
		$('#exploreModal').modal('show');
	});

  $('.pop').on('click', function() {
    contentURL = $(this).find('img').attr('src');
    $('#imagepreview').attr('src', contentURL);
    $('#homeModal').modal('show');
  });

  $('.fb-share-button').on('click', function() {
    $(this).attr('data-href', contentURL);
  })


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
              $('#uploadModal').modal('show');
          }

          reader.readAsDataURL(input.files[0]);
      }
  }


  // report content pop up 
  $('#contact').click(function() {
    $('#contactForm').fadeToggle();
  })
  $(document).mouseup(function (e) {
    var container = $("#contactForm");

    if (!container.is(e.target) // if the target of the click isn't the container...
        && container.has(e.target).length === 0) // ... nor a descendant of the container
    {
        container.fadeOut();
    }
  });
})


