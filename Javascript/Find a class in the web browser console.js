(function () {
  'use strict';
  
  var menuitem = document.querySelectorAll('[role="menuitem"]');
  
  [].forEach.call(menuitem, function (el) {
    
    // Default mouse click event
    el.addEventListener('click', buttonClick, false);
    
    // Handle 'space bar' and 'enter' click events,
    // expected behaviour for an actual <button> element
    el.addEventListener('keydown', function (e) {      
      if (e.which === 32 || e.which === 13) {
        e.preventDefault();
        buttonClick();
      }
    }, false);
    
  });
  
  function buttonClick() {    
    // Do something useful on each click
    console.log('Clicked!');
  }
  
})();
