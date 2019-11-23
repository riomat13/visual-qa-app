var closebtns = document.getElementsByClassName("close");

for (i = 0; i < closebtns.length; i++) {
    closebtns[i].addEventListener("click", function(){
        this.parentElement.style.display = 'none';
    });
}