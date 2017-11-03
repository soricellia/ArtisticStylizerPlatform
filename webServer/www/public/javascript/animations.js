// global varialbes
var bounce = 150;
var delay = 0;
var nameLeft = 10;


$(document).ready(function(){


	$('#left-stick').delay(200).animate({
		left: '-50px'
	}, {
		step: function(currentLeft) { 
	        var letter;
	        console.log(currentLeft);
	        $('.left-letter').each(function(n,item){
	        	letter = $(item);
	        	letterPostion = letter.position().left
	        	console.log(letterPostion);
	        	if (letterPostion > currentLeft) {
	        		$(letter).css({
	        			opacity: 1
	        		})
	        	}
	        })	
	    } 		
	},500);


	$('#right-stick').delay(200).animate({
		left: '1500px'
	}, {
		step: function(currentLeft) { 
	        var letter;
	        console.log(currentLeft);
	        $('.right-letter').each(function(n,item){
	        	letter = $(item);
	        	letterPostion = letter.position().left
	        	console.log(letterPostion);
	        	if (letterPostion < currentLeft) {
	        		$(letter).css({
	        			opacity: 1
	        		})
	        	}
	        })	
	    } 		
	},500);


})		