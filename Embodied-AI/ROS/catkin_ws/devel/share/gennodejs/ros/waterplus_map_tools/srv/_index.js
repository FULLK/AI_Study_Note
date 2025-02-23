
"use strict";

let GetWaypointByName = require('./GetWaypointByName.js')
let SaveWaypoints = require('./SaveWaypoints.js')
let GetWaypointByIndex = require('./GetWaypointByIndex.js')
let AddNewWaypoint = require('./AddNewWaypoint.js')
let GetNumOfWaypoints = require('./GetNumOfWaypoints.js')
let GetChargerByName = require('./GetChargerByName.js')

module.exports = {
  GetWaypointByName: GetWaypointByName,
  SaveWaypoints: SaveWaypoints,
  GetWaypointByIndex: GetWaypointByIndex,
  AddNewWaypoint: AddNewWaypoint,
  GetNumOfWaypoints: GetNumOfWaypoints,
  GetChargerByName: GetChargerByName,
};
