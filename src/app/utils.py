head_html = """
<meta name="viewport" 
      content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
"""

css_code = """
input {
    font-size: 16px !important;
}
input[type="number"] {
    inputmode: numeric;
    pattern: "[0-9]*";
}
#fullwidth-checkgroup .gr-checkgroup {
    width: 100%;
}
footer {
    display: none !important;
}

.loader {
    width: 48px;
    height: 48px;
    border: 5px solid #FFF;
    border-bottom-color: transparent;
    border-radius: 50%;
    display: inline-block;
    box-sizing: border-box;
    animation: rotation 1s linear infinite;
    }

    @keyframes rotation {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
    } 
"""

spinner_html = """
<div style="
  display: flex; 
  justify-content: center; 
  align-items: center; 
  height: 10vh; 
  width: 100%;
">
  <span class='loader'></span>
</div>
"""