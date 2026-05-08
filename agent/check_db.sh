#!/bin/bash

DB_URL="postgresql://sidya:sidya2026@localhost/sidya_transport"

echo ""
echo "==============================="
echo "   ENTREPRISES"
echo "==============================="
psql $DB_URL -c "SELECT id, nom FROM entreprises ORDER BY id;"

echo ""
echo "==============================="
echo "   VOYAGES"
echo "==============================="
psql $DB_URL -c "
SELECT 
  v.id,
  v.date,
  v.num_camion,
  v.tonnage,
  e.nom AS entreprise,
  v.prix_camion_par_tonne AS prix_camion,
  v.prix_client_par_tonne AS prix_client,
  ROUND((v.tonnage * v.prix_client_par_tonne)::numeric, 0) AS montant_client,
  ROUND((v.tonnage * (v.prix_client_par_tonne - v.prix_camion_par_tonne))::numeric, 0) AS benefice
FROM voyages v
JOIN entreprises e ON e.id = v.entreprise_id
ORDER BY v.date DESC, v.id DESC;
"

echo ""
echo "==============================="
echo "   TOTAL PAR ENTREPRISE"
echo "==============================="
psql $DB_URL -c "
SELECT 
  e.nom AS entreprise,
  COUNT(v.id) AS nb_voyages,
  SUM(v.tonnage) AS total_tonnes,
  ROUND(SUM(v.tonnage * v.prix_client_par_tonne)::numeric, 0) AS total_client,
  ROUND(SUM(v.tonnage * v.prix_camion_par_tonne)::numeric, 0) AS total_camions,
  ROUND(SUM(v.tonnage * (v.prix_client_par_tonne - v.prix_camion_par_tonne))::numeric, 0) AS benefice_net
FROM voyages v
JOIN entreprises e ON e.id = v.entreprise_id
GROUP BY e.nom
ORDER BY benefice_net DESC;
"
