# -*- coding: utf-8 -*-
"""
Enhanced Movie Selector
Author: pineaulo (Enhanced by AI)
Version: 3.0 (Interactive UI)
"""

import sys
import os
import requests
import pandas as pd
from io import BytesIO

# Try to import tabulate for pretty printing
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# --- Configuration ---
SHEET_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQ-I3jNSGm4lQ4woB_RduhnQ-Y_23h-okao2BBRxGe4ESRfOatDNtl4ibZcN8HETMPDmvN_FuCzeKrC/pub?gid=0&single=true&output=csv'

# --- Utility Functions ---
def clear_screen():
    """Clears the console screen based on OS."""
    os.system('cls' if os.name == 'nt' else 'clear')

def wait_for_user():
    """Pauses execution so the user can read the results."""
    input("\n👉 Appuyez sur [Entrée] pour revenir au menu...")

# --- Core Functions ---
def get_data():
    """Fetches data from Google Sheets."""
    print("📡 Connexion à Google Sheets en cours...")
    
    try:
        r = requests.get(SHEET_URL, timeout=15)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Erreur de connexion Internet : {e}")
        sys.exit(1)

    try:
        df = pd.read_csv(BytesIO(r.content), index_col=0) 
    except Exception as e:
        print(f"\n❌ Erreur CSV : {e}")
        sys.exit(1)

    # Clean Data
    df.columns = df.columns.str.strip() 
    required_cols = ['Statut', 'Catégorie', 'Titre']
    
    # Check columns
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Colonnes manquantes. Colonnes du fichier : {list(df.columns)}")
        sys.exit(1)

    for col in required_cols:
        df[col] = df[col].astype(str).str.strip()

    df_unseen = df[df['Statut'] == 'Pas encore vu'].copy()

    if df_unseen.empty:
        print("\n⚠️ Aucun film 'Pas encore vu' trouvé.")
        sys.exit(0)

    print(f"✅ {len(df_unseen)} films disponibles.")
    # Short pause to let user see the connection success
    import time
    time.sleep(1.5) 
    return df_unseen

def display_results(selection, title):
    """Prints the selected movies."""
    print(f"\n🍿 --- {title} --- 🍿\n")
    
    display_cols = ['Catégorie', 'Titre']
    if 'Année' in selection.columns:
        display_cols.append('Année')

    if selection.empty:
        print("Aucune sélection.")
    elif HAS_TABULATE:
        print(tabulate(selection[display_cols], headers='keys', tablefmt='fancy_grid', showindex=False))
    else:
        for idx, row in selection.iterrows():
            print(f"• [{row['Catégorie']}] : {row['Titre']}")

def main_menu():
    df = get_data()

    while True:
        clear_screen() # Clears screen at start of loop
        
        print("================================")
        print("      🎬 SÉLECTEUR DE FILMS     ")
        print("================================")
        print("1. ⚖️  Équilibré (1 par catégorie)")
        print("2. 🎲  Aléatoire (3 au hasard)")
        print("3. 🎯  Roulette Russe (1 gagnant)")
        print("q. Quitter")
        
        choice = input("\nVotre choix : ").strip().lower()

        if choice == '1':
            try:
                selection = df.groupby('Catégorie').apply(lambda x: x.sample(1)).reset_index(drop=True)
                display_results(selection, "SÉLECTION PAR CATÉGORIE")
            except Exception as e:
                print(f"Erreur : {e}")
            wait_for_user() # <--- Holds the screen

        elif choice == '2':
            count = min(3, len(df))
            selection = df.sample(n=count)
            display_results(selection, "MÉLANGE ALÉATOIRE")
            wait_for_user() # <--- Holds the screen

        elif choice == '3':
            winner = df.sample(n=1).iloc[0]
            print("\n" + "*"*40)
            print(f"🏆 GAGNANT : {winner['Titre'].upper()}")
            print(f"📂 Genre   : {winner['Catégorie']}")
            print("*"*40)
            wait_for_user() # <--- Holds the screen

        elif choice == 'q':
            print("\nBon visionnage ! 👋")
            break
        
        else:
            print("\n❌ Choix invalide.")
            import time
            time.sleep(1) # Small pause before reloading menu

if __name__ == "__main__":
    main_menu()
