var map;
var groupColour = ['#1a9850','#91cf60','#d73027','#fc8d59','#fee08b','#d9ef8b'];
var ratingColour = ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026'];
function initMap() {
	map = new google.maps.Map(document.getElementById('map'), {
		zoom: 11,
		center: new google.maps.LatLng(33.38, -112.071540),
		//mapTypeId: 'terrain'
		});

	function circle(colour) {
        return {
          path: google.maps.SymbolPath.CIRCLE,
          fillColor: colour,
          scale: 1,
          strokeColor: colour,
          strokeWeight: .5
        };
      }

    function circleCentroid(colour) {
        return {
          path: google.maps.SymbolPath.CIRCLE,
          fillColor: colour,
          scale: 3,
          strokeColor: 'black',
          strokeWeight: .5
        };
      }

    function getPoints(){
    	var points = [];
    	for (var i = 0; i < locations.length; i++) {
			var latlng = new google.maps.LatLng(locations[i][0],locations[i][1]);
			var weightedLoc = {
				location: latLng,
				weight: locations[3]
			};
			points.push(weightedLoc);
		}
		return points;
    }

	for (var i = 0; i < centroids.length; i++) {
		var colour = groupColour[i];
		var latLng = new google.maps.LatLng(centroids[i][0],centroids[i][1]);
		var marker = new google.maps.Marker({
			icon: circleCentroid(colour),
			position: latLng,
			map: map
		});
	}

	var heatmap = new google.maps.visualization.HeatmapLayer({
          data: getPoints(),
          //dissipating: false,
          map: map
        }); 
}
