/**
 * FacilityMap - Embedded map component showing user location and nearby healthcare facilities
 * Uses Leaflet (free, no API key required) with OpenStreetMap tiles
 * 
 * Props:
 * - mapData: Object containing user_location, facilities, center, zoom
 * - onFacilityClick: Callback when a facility marker is clicked
 * - height: Map height (default: "400px")
 */

import React, { useEffect, useRef, useState } from 'react';

// Leaflet CSS should be included in your HTML:
// <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />

const FacilityMap = ({ mapData, onFacilityClick, height = "400px" }) => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersRef = useRef([]);
  const [selectedFacility, setSelectedFacility] = useState(null);

  useEffect(() => {
    // Dynamically load Leaflet if not already loaded
    if (!window.L) {
      const script = document.createElement('script');
      script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
      script.onload = () => initMap();
      document.head.appendChild(script);

      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
      document.head.appendChild(link);
    } else {
      initMap();
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (mapInstanceRef.current && mapData) {
      updateMarkers();
    }
  }, [mapData]);

  const initMap = () => {
    if (!mapRef.current || mapInstanceRef.current) return;
    if (!mapData?.center) return;

    const L = window.L;
    
    // Initialize map
    const map = L.map(mapRef.current).setView(
      [mapData.center.latitude, mapData.center.longitude],
      mapData.zoom || 11
    );

    // Add OpenStreetMap tiles (free, no API key)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    mapInstanceRef.current = map;
    updateMarkers();
  };

  const updateMarkers = () => {
    if (!mapInstanceRef.current || !mapData) return;

    const L = window.L;
    const map = mapInstanceRef.current;

    // Clear existing markers
    markersRef.current.forEach(marker => map.removeLayer(marker));
    markersRef.current = [];

    // Custom icon for user location (blue)
    const userIcon = L.divIcon({
      className: 'custom-marker',
      html: `
        <div style="
          background: #2563eb;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          border: 3px solid white;
          box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        "></div>
      `,
      iconSize: [20, 20],
      iconAnchor: [10, 10],
    });

    // Add user location marker
    if (mapData.user_location?.latitude && mapData.user_location?.longitude) {
      const userMarker = L.marker(
        [mapData.user_location.latitude, mapData.user_location.longitude],
        { icon: userIcon }
      ).addTo(map);
      
      userMarker.bindPopup(`
        <div style="text-align: center;">
          <strong>Your Location</strong><br/>
          ${mapData.user_location.zipcode || ''}
        </div>
      `);
      
      markersRef.current.push(userMarker);
    }

    // Add facility markers
    if (mapData.facilities?.length > 0) {
      mapData.facilities.forEach((facility, index) => {
        if (!facility.latitude || !facility.longitude) return;

        // Custom numbered icon for facilities (teal/green like Turquoise Health)
        const facilityIcon = L.divIcon({
          className: 'custom-marker',
          html: `
            <div style="
              background: #0d9488;
              color: white;
              width: 28px;
              height: 28px;
              border-radius: 50%;
              border: 2px solid white;
              box-shadow: 0 2px 6px rgba(0,0,0,0.3);
              display: flex;
              align-items: center;
              justify-content: center;
              font-weight: bold;
              font-size: 12px;
            ">${facility.index || index + 1}</div>
          `,
          iconSize: [28, 28],
          iconAnchor: [14, 14],
        });

        const marker = L.marker(
          [facility.latitude, facility.longitude],
          { icon: facilityIcon }
        ).addTo(map);

        // Create popup content
        const priceText = facility.price 
          ? `$${facility.price.toLocaleString()}`
          : (facility.estimated_range || 'Price not available');
        
        const popupContent = `
          <div style="min-width: 200px;">
            <strong style="color: #0d9488; font-size: 14px;">${facility.name}</strong>
            <br/>
            <span style="color: #666; font-size: 12px;">${facility.address || ''}</span>
            <br/>
            <div style="margin-top: 8px; display: flex; justify-content: space-between; align-items: center;">
              <span style="font-size: 12px; color: #888;">
                ${facility.distance_miles ? `${facility.distance_miles.toFixed(1)} mi` : ''}
              </span>
              <span style="font-weight: bold; color: #0d9488; font-size: 16px;">
                ${priceText}
              </span>
            </div>
          </div>
        `;

        marker.bindPopup(popupContent);
        
        marker.on('click', () => {
          setSelectedFacility(facility);
          if (onFacilityClick) {
            onFacilityClick(facility);
          }
        });

        markersRef.current.push(marker);
      });

      // Fit bounds to show all markers
      const allCoords = [
        [mapData.user_location?.latitude, mapData.user_location?.longitude],
        ...mapData.facilities.map(f => [f.latitude, f.longitude])
      ].filter(c => c[0] && c[1]);

      if (allCoords.length > 1) {
        const bounds = L.latLngBounds(allCoords);
        map.fitBounds(bounds, { padding: [50, 50] });
      }
    }
  };

  const handleSearchWhenMoving = (e) => {
    // This could trigger a new search when the map is moved
    // For now, just a placeholder
    console.log('Search when moving:', e.target.checked);
  };

  if (!mapData) {
    return (
      <div style={{
        height,
        background: '#f3f4f6',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: '8px',
        color: '#6b7280'
      }}>
        Loading map...
      </div>
    );
  }

  return (
    <div style={{ position: 'relative' }}>
      {/* Search when moving checkbox */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        zIndex: 1000,
        background: 'white',
        padding: '8px 12px',
        borderRadius: '4px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        fontSize: '13px',
        display: 'flex',
        alignItems: 'center',
        gap: '6px'
      }}>
        <input 
          type="checkbox" 
          id="searchWhenMoving"
          onChange={handleSearchWhenMoving}
          style={{ accentColor: '#0d9488' }}
        />
        <label htmlFor="searchWhenMoving">Search when moving</label>
      </div>

      {/* Map container */}
      <div 
        ref={mapRef} 
        style={{ 
          height, 
          width: '100%',
          borderRadius: '8px',
          overflow: 'hidden'
        }} 
      />

      {/* Price labels on map (optional overlay) */}
      {mapData.facilities?.length > 0 && (
        <div style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          zIndex: 1000,
          background: 'white',
          padding: '8px',
          borderRadius: '4px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          fontSize: '11px',
          color: '#666'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: '#2563eb'
            }}></div>
            <span>Your location</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
            <div style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: '#0d9488'
            }}></div>
            <span>Healthcare facilities</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default FacilityMap;
