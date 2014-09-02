define(["utils/utils","mvc/ui/ui-select-default","mvc/ui/ui-slider","mvc/ui/ui-checkbox","mvc/ui/ui-radiobutton","mvc/ui/ui-button-menu","mvc/ui/ui-modal"],function(e,h,k,i,b,r,c){var q=Backbone.View.extend({optionsDefault:{url:"",cls:""},initialize:function(s){this.options=e.merge(s,this.optionsDefault);this.setElement(this._template(this.options))},_template:function(s){return'<img class="ui-image '+s.cls+'" src="'+s.url+'"/>'}});var l=Backbone.View.extend({optionsDefault:{title:"",cls:""},initialize:function(s){this.options=e.merge(s,this.optionsDefault);this.setElement(this._template(this.options))},title:function(s){this.$el.html(s)},_template:function(s){return'<label class="ui-label '+s.cls+'">'+s.title+"</label>"},value:function(){return options.title}});var d=Backbone.View.extend({optionsDefault:{floating:"right",icon:"",tooltip:"",placement:"bottom",title:"",cls:""},initialize:function(s){this.options=e.merge(s,this.optionsDefault);this.setElement(this._template(this.options));$(this.el).tooltip({title:s.tooltip,placement:"bottom"})},_template:function(s){return'<div><span class="fa '+s.icon+'" class="ui-icon"/>&nbsp;'+s.title+"</div>"}});var g=Backbone.View.extend({optionsDefault:{id:null,title:"",floating:"right",cls:"btn btn-default",icon:""},initialize:function(s){this.options=e.merge(s,this.optionsDefault);this.setElement(this._template(this.options));$(this.el).on("click",s.onclick);$(this.el).tooltip({title:s.tooltip,placement:"bottom"})},_template:function(s){var t='<button id="'+s.id+'" type="submit" style="float: '+s.floating+';" type="button" class="ui-button '+s.cls+'">';if(s.icon){t+='<i class="icon fa '+s.icon+'"></i>&nbsp;'}t+=s.title+"</button>";return t}});var o=Backbone.View.extend({optionsDefault:{id:null,title:"",floating:"right",cls:"icon-btn",icon:"",tooltip:"",onclick:null},initialize:function(s){this.options=e.merge(s,this.optionsDefault);this.setElement(this._template(this.options));$(this.el).on("click",s.onclick);$(this.el).tooltip({title:s.tooltip,placement:"bottom"})},_template:function(s){var t="";if(s.title){t="width: auto;"}var u='<div id="'+s.id+'" style="float: '+s.floating+"; "+t+'" class="ui-button-icon '+s.cls+'">';if(s.title){u+='<div class="button"><i class="icon fa '+s.icon+'"/>&nbsp;<span class="title">'+s.title+"</span></div>"}else{u+='<i class="icon fa '+s.icon+'"/>'}u+="</div>";return u}});var p=Backbone.View.extend({optionsDefault:{title:"",cls:""},initialize:function(s){this.options=e.merge(s,this.optionsDefault);this.setElement(this._template(this.options));$(this.el).on("click",s.onclick)},_template:function(s){return'<div><a href="javascript:void(0)" class="ui-anchor '+s.cls+'">'+s.title+"</a></div>"}});var a=Backbone.View.extend({optionsDefault:{message:"",status:"info",persistent:false},initialize:function(s){this.options=e.merge(s,this.optionsDefault);this.setElement("<div></div>")},update:function(t){this.options=e.merge(t,this.optionsDefault);if(t.message!=""){this.$el.html(this._template(this.options));this.$el.find(".alert").append(t.message);this.$el.fadeIn();if(!t.persistent){var s=this;window.setTimeout(function(){if(s.$el.is(":visible")){s.$el.fadeOut()}else{s.$el.hide()}},3000)}}else{this.$el.fadeOut()}},_template:function(s){return'<div class="ui-message alert alert-'+s.status+'"/>'}});var f=Backbone.View.extend({optionsDefault:{onclick:null,searchword:""},initialize:function(t){this.options=e.merge(t,this.optionsDefault);this.setElement(this._template(this.options));var s=this;if(this.options.onclick){this.$el.on("submit",function(v){var u=s.$el.find("#search");s.options.onclick(u.val())})}},_template:function(s){return'<div class="ui-search"><form onsubmit="return false;"><input id="search" class="form-control input-sm" type="text" name="search" placeholder="Search..." value="'+s.searchword+'"><button type="submit" class="btn search-btn"><i class="fa fa-search"></i></button></form></div>'}});var n=Backbone.View.extend({optionsDefault:{value:"",type:"text",placeholder:"",disabled:false,visible:true,cls:""},initialize:function(t){this.options=e.merge(t,this.optionsDefault);this.setElement(this._template(this.options));if(this.options.disabled){this.$el.prop("disabled",true)}if(!this.options.visible){this.$el.hide()}var s=this;this.$el.on("input",function(){if(s.options.onchange){s.options.onchange(s.$el.val())}})},value:function(s){if(s!==undefined){this.$el.val(s)}return this.$el.val()},_template:function(s){return'<input id="'+s.id+'" type="'+s.type+'" value="'+s.value+'" placeholder="'+s.placeholder+'" class="ui-input '+s.cls+'">'}});var j=Backbone.View.extend({optionsDefault:{value:"",type:"text",placeholder:"",disabled:false,visible:true,cls:""},initialize:function(t){this.options=e.merge(t,this.optionsDefault);this.setElement(this._template(this.options));if(this.options.disabled){this.$el.prop("disabled",true)}if(!this.options.visible){this.$el.hide()}var s=this;this.$el.on("input",function(){if(s.options.onchange){s.options.onchange(s.$el.val())}})},value:function(s){if(s!==undefined){this.$el.val(s)}return this.$el.val()},_template:function(s){return'<textarea id="'+s.id+'" class="ui-textarea '+s.cls+'" rows="5"></textarea>'}});var m=Backbone.View.extend({optionsDefault:{value:""},initialize:function(s){this.options=e.merge(s,this.optionsDefault);this.setElement(this._template(this.options))},value:function(s){if(s!==undefined){this.$el.val(s)}return this.$el.val()},_template:function(s){return'<hidden id="'+s.id+'" value="'+s.value+'"/>'}});return{Anchor:p,Button:g,ButtonIcon:o,ButtonMenu:r,Icon:d,Image:q,Input:n,Label:l,Message:a,Modal:c,RadioButton:b,Checkbox:i,Searchbox:f,Select:h,Textarea:j,Hidden:m,Slider:k}});