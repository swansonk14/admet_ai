// this file contains a set of standard scripts used in many pages

/*
============
open window with focus
============
// windows opened with window.open function and a window name, receive focus only when a window with the given name did not yet exist
// If a window with the given name already exists, the window will load in the background
// this function will bring the focus to the window when opening it again 
*/
function openFocussed (winHref, winName) {
	var tmpWindow = window.open(winHref, winName);
	tmpWindow.focus();
}
/*
============
	selectbox navigation
============
*/
// typing in the reference input field selects the first matching entry in the selectbox
// item should match from start of entry label
function matchFirst(srcField, tarSelect){
	for (var i=0;i<tarSelect.length;i++){
		if(tarSelect.options[i].text.toLowerCase().indexOf(srcField.value.toLowerCase())==0){
			tarSelect.selectedIndex=i;
			return;
		}
	}
}

// typing in the reference field hides non-matching entries
// item can match in any position
function limitList(srcField,tarSelect){
// below line creates a case insensitive regular expression pattern based on the contents of srcField 
	var srcTest = eval('/' + srcField.value + '/i');
	var firstmatch = true;
	var i;
	var iEnd = tarSelect.length;
	for (i=0;i < iEnd; i++){
		if (!srcField.value.length) {
			tarSelect.options[i].style.display = 'block';
			tarSelect.options[i].selected = false;
		} else if (srcTest.test(tarSelect.options[i].title)) {
			tarSelect.options[i].style.display = 'block';
			tarSelect.options[i].selected = firstmatch;
			if (firstmatch) {firstmatch = false;}
		} else {
			tarSelect.options[i].style.display = 'none';
			tarSelect.options[i].selected = false;
		}
	}
}

/*
============
	form submission
============
*/
// store active form and field name in trigger field before submitting a form
// useful for detecting which form triggered the submission (onblur autosubmission))
function autoSubmit(refField) {
	if (refField.form.elements.namedItem('trigger')) {refField.form.trigger.value=refField.form.name + ':' + refField.name;}
	refField.form.submit();
}

// store old variable value in hidden field, e.g. when primary key is text field
// old forms use field 'new_val' for new value, new forms use field 'add'
function updateField(refField) {
	refField.form.orig_val.value=refField.value;
	refField.form.new_val.value=refField.value;
}

// read maintenance list values and write to appropriate fields
function updateMaintField(selectField) {
	if (document.getElementById(selectField.form.name+':up')) {selectField.form.up.disabled = false;}
	if (document.getElementById(selectField.form.name+':down')) {selectField.form.down.disabled = false;}
	selectField.form.idval.value = selectField.value;
	selectField.form.labelval.value = selectField[selectField.selectedIndex].innerHTML;
	if (document.getElementById(selectField.form.name+':otherval')) {selectField.form.otherval.value = selectField[selectField.selectedIndex].title;}
	if (document.getElementById(selectField.form.name+':classval')) {
		selectField.form.classval.value = selectField[selectField.selectedIndex].className;
		selectField.form.classval.className = selectField[selectField.selectedIndex].className;
		styles=/([\w]+)\s*([\w]*)/.exec(selectField[selectField.selectedIndex].className);
		if (styles[1]) {selectField.form.fgcol.value = styles[1];} else {selectField.form.fgcol.value='black';}
		if (styles[2]) {selectField.form.bgcol.value = styles[2];} else {selectField.form.bgcol.value='whiteback';}
	}
}

// read stylecolours and write to classfield
function readStyleColours(styleForm) {
	styledef = styleForm.fgcol.value+' '+styleForm.bgcol.value;
	styleForm.classval.value = styledef;
	styleForm.classval.className = styledef;
}
/*
============
	field content verification
============
*/

function checkDate(refField){
	if (refField.value.length>0){
		re=/^(\d{4})-?(\d{2})-?(\d{2})$/;
		if (re.test(refField.value)){
			refField.value=refField.value.replace(re,"$1-$2-$3");
			refField.className="whiteback";
		} else {
			re=/^(\d{2})-(\d{2})-(\d{4})$/;
			if (re.test(refField.value)){
				refField.value=refField.value.replace(re, "$3-$2-$1");
				refField.className="whiteback";
			} else {
/*
		the below regular expression does not work, debug needed
				re=/^(\d{2})/(\d{2})/(\d{4})$/;
		try this one instead
				re=/^(\d{2})\D(\d{2})\D(\d{4})$/;
				if (re.test(refField.value)){
					refField.value=refField.value.replace(re, "$3-$2-$1");
					refField.className="whiteback";
				} else {
*/					refField.className="redback";
//				}
			}
		}
	} else {
		refField.className="whiteback";
	}
}

function checkInt(refField){
	if (refField.value.length>0){
		if (parseInt(refField.value)==refField.value){
			refField.className="whiteback";
			refField.value=parseInt(refField.value);
		} else {
			refField.className="redback";
		}
	} else {
		refField.className="whiteback";
	}
}

function checkIntsubmit(refField){
	checkInt(refField);
	if (refField.className=="whiteback"){autoSubmit(refField);}
}

function checkNum(refField){
	if (refField.value.length>0){
		if (isNaN(refField.value)){
			refField.className="redback";
		} else {
			refField.className="whiteback";
		}
	} else {
		refField.className="whiteback";
	}
}

function checkNumsubmit(refField){
	checkNum(refField);
	if (refField.className=="whiteback"){autoSubmit(refField);}
}

/*
============
	toggle collapsable divs triggered by clicking a designated span
============
*/

function divToggle(srcSPAN,tarID) {
	var tarDIV = document.getElementById(tarID);
	if (srcSPAN.className=='show_detail') {
		srcSPAN.className='hide_detail';
		tarDIV.className='collapsed';
	} else {
		srcSPAN.className='show_detail';
		tarDIV.className='';
	}
}

/*
============
	(un)check all checkboxes whose name starts with a specific term
	variable boxName is starting portion of checkboxes to set
	variable boxState is either 'true' or 'false' (without quotes)
============
*/

function setBoxes(boxName,boxState) {
	var tarBoxes = document.querySelectorAll("input[name^=" + boxName + "]");
	for (var i = 0; i < tarBoxes.length; ++i) {
		if (tarBoxes[i].type=='checkbox') {tarBoxes[i].checked = boxState;}
	}
}