exports.build = function(title, pagetitle, content){
	return [

    '<!DOCTYPE HTML>',
    '<html lang="en">',
    '<head>',
    	'<title>ASP</title>',
      '<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">',
      '<link href="https://fonts.googleapis.com/css?family=Libre+Franklin" rel="stylesheet">',
      '<script type = "text/javascript" src = "https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>',
      '<script type = "text/javascript" src = "https://ajax.googleapis.com/ajax/libs/jqueryui/1.11.3/jquery-ui.min.js"></script>',
      '<link href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">',
      '<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script>',
      '<script src="https://unpkg.com/masonry-layout@4/dist/masonry.pkgd.js"></script>',
      '<script src="https://unpkg.com/imagesloaded@4/imagesloaded.pkgd.min.js"></script>',
      '<link rel="stylesheet" type="text/css" href="assets/css/style.css">',
    	'<script type="text/javascript" src="assets/javascript/home-grid.js"></script>',
    '</head>',
    '<body>',
    	'<nav class="navbar navbar-inverse">',
      	'<div class="container-fluid">',
        	'<div class="navbar-header">',
          	'<button type="button" class="navbar-toggle" data-toggle="collapse" data-target="#myNavbar">',
            	'<span class="icon-bar"></span>',
            	'<span class="icon-bar"></span>',
            	'<span class="icon-bar"></span>',
          	'</button>',
          	'<a class="navbar-brand" href="home">ASP</a>',
        	'</div>',
        	'<div class="collapse navbar-collapse" id="myNavbar">',
          	'<ul class="nav navbar-nav navbar-right">',
              '<li><a href="about"><span></span>About</a><li>',
          	'</ul>',
        	'</div>',
      	'</div>',
    	'</nav>',

      '<div class="container profile-container">',
          '<div class="row">',
            '<div class="col-md-2 col-md-offset-2">',
              '<img src="assets/images/pip.PNG" alt="Profile Picture" height="120" width="120">',
            '</div>',
    				'<div class="col-md-8">',
    					'<h3>d_trump44</h3>',
    				'</div>',
          '</div>',
      '</div>',

    	'<div class="account-content">',
    	  '<div class="grid">',
    	    '<div class="grid-sizer"></div>',
    	    '<div class="grid-item">',
    	      '<img src="assets/images/dinah.JPG" />',
    	    '</div>',
    	    '<div class="grid-item">',
    	      '<img src="assets/images/rowing.JPG" />',
    	    '</div>',
    	    '<div class="grid-item">',
    	      '<img src="assets/images/rowing2.jpg" />',
    	    '</div>',
    	    '<div class="grid-item">',
    	      '<img src="assets/images/dog.JPG" />',
    	    '</div>',
    	    '<div class="grid-item">',
    	      '<img src="assets/images/cape.JPG" />',
    	    '</div>',
    	    '<div class="grid-item">',
    	      '<img src="assets/images/pip.PNG" />',
    	    '</div>',
    	    '<div class="grid-item">',
    	      '<img src="assets/images/snow.JPG" />',
    	    '</div>',
    	    '<div class="grid-item">',
    	      '<img src="assets/images/compound.PNG" />',
    	    '</div>',
    	    '<div class="grid-item">',
    	     ' <img src="assets/images/food.JPG" />',
    	    '</div>',
    	  '</div>',
    	'</div>',


      '<div class="footer-container">',
            '<a class="footer-content" mailto="ebinizerlop@gmail.com">Email Us</a>',
            '<span class ="footer-content" >&copy; Copyright 1738. No Rights Reserved</span>',
      '</div>',

    '</body>']

    .join('')

};
