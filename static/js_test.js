document.addEventListener('DOMContentLoaded', function(){
	var file_upload = document.getElementsByClassName('file_upload')[0];
	
	file_upload.addEventListener('change', function(){
		this.nextElementSibling.setAttribute('data-file', this.value);
	});
	
});