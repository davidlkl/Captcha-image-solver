$("body").on('click','#btnGenerate',function(){
	$this =$(this);
	$this.prop('disabled',true);
	$.getJSON('/generate_and_predict', {
			}, function(data) {
		   $('.subtitle').show();
		   document.getElementById("captcha").src = data.image_url;
		   document.getElementById("prediction").innerHTML = data.prediction;
		   document.getElementById("target").innerHTML = data.target;
		   $this.prop('disabled',false);	
		  });
});