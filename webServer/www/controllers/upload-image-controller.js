/**
	This is just a little demo of what we can do with node.js
	we first get the data we need (in this case wer're requiring
	'../model/test-data' but in the future we can get data from a database)
	after we get the data we brake it down into list items to be printed to the
	user.
**/

//things we need -- the test data and the homepage html template
var User = require('../model/user');
var Profile = require('../model/profile');
var multer = require('multer');
var upload = multer({dest: 'tmp/'})
var PythonShell = require('python-shell');

/*
exports.post = function(req, res){
	var form = new formidable.IncomingForm();
	form.parse(req, function(err, fields, files){
		var oldpath = files.filetoupload.path;
		var newpath = '/home/mike/' + files.filetoupload.name;
		fs.rename(oldpath, newpath, function(err){
			if (err) throw err;
			res.write('files uploaded and moved!');
			res.end();
		});
	});
}
*/

var storage = multer.diskStorage({
	destination: function(req, res, callback){
		callback(null, 'public/tmp/');
	},

	filename: function(req, file, callback){
		console.log(file);
		callback(null, file.originalname)
	}
});

var upload = multer({storage: storage}).array('photo', 2);

exports.post = function(req, res){
	// upload(req, res, function(err){
	// 	if(err){
	// 		console.log('error occured');
	// 		return;
	// 	}
	// 	// req.files is an ojbect where fieldname is the key and value is the array of files
	// 	console.log(req.files);
	// 	console.log('photo uploaded');
		var options = {
			pythonPath: '/usr/bin/python3',
		        scriptPath: '/home/mike/ArtisticStylizerPlatform/gpuServer/AS/src',
		        args: ['public/images/dog.jpg', 'public/default_styles/deja_vu.jpg', '/home/morgan/ArtisticStylizerPlatform/webServer/www/public/tmp', 256, 512]
		};
		try{
			PythonShell.run('inference_master.py', options, function(err){
				if (err) throw err;
				
				// load the user first
				var currentUser = req.user;
				console.log("current user: ", currentUser);

				Profile.getProfile(currentUser.userid, function(err, result) {
					if (err) throw err
					
					var userProfile = result;
					
					return res.render("../views/home.ejs", { user : req.user });
				})
			});
		}
		catch(err){
			console.log(err);
		}
	// });

};