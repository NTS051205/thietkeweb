let images = [
    "url('311408098_638948004608909_191367019285130494_n.jpg')", 
    "url('410325686_887079483129092_2221573659796163453_n.jpg')", 
    "url('433255139_943398900830483_682642798376790520_n.jpg')", 
    "url('z4859380210844_e6dd17b63accb08f72539a1a4bff88bc.jpg')", 
    "url('z5777600531309_0740dd7e10a9e7c872a6db5765597bec.jpg')"
];
let i = 0;
let n = images.length;

function changeImage() {
    let image = images[i];
    document.body.style.backgroundImage = image;
    document.body.style.backgroundSize = '100%';
    document.body.style.backgroundPosition = 'cover';
    document.body.style.backgroundRepeat = 'no-repeat';
    document.body.style.backgroundColor = 'black';
    document.body.style.backgroundAttachment = 'fixed';
    i++;
    if(i == n) {
        i = 0;
    }
}
setInterval(changeImage, 1000);