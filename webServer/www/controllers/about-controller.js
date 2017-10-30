

	after we get the data we brake it down into list items to be printed to the
	user.

**/

exports.get = function(req, res) {

	return res.render("../views/about.ejs", { user : req.user });

};
