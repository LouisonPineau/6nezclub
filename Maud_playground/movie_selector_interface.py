# -*- coding: utf-8 -*-
"""
Movie Selector - GUI (Tkinter)
Author: pineaulo (Enhanced by AI)
"""

import os
import time
import requests
import pandas as pd
from io import BytesIO
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox


# --- Configuration ---
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ-I3jNSGm4lQ4woB_RduhnQ-Y_23h-okao2BBRxGe4ESRfOatDNtl4ibZcN8HETMPDmvN_FuCzeKrC/pub?gid=0&single=true&output=csv"


# --- Data / Core logic ---
def fetch_data() -> pd.DataFrame:
    """Fetches data from Google Sheets and returns unseen movies dataframe."""
    try:
        r = requests.get(SHEET_URL, timeout=15)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erreur de connexion Internet : {e}") from e

    try:
        df = pd.read_csv(BytesIO(r.content))
    except Exception as e:
        raise RuntimeError(f"Erreur CSV : {e}") from e

    # Clean & validate
    df.columns = df.columns.str.strip()
    required_cols = ["Statut", "Catégorie", "Titre"]
    if not all(col in df.columns for col in required_cols):
        raise RuntimeError(
            f"Colonnes manquantes. Colonnes trouvées : {list(df.columns)}"
        )

    for col in required_cols:
        df[col] = df[col].astype(str).str.strip()

    # Optional year column
    if "Année" in df.columns:
        # try numeric
        df["Année"] = pd.to_numeric(df["Année"], errors="coerce")

    df_unseen = df[df["Statut"] == "Pas encore vu"].copy()
    if df_unseen.empty:
        raise RuntimeError("Aucun film 'Pas encore vu' trouvé.")

    return df_unseen


def available_categories(df: pd.DataFrame) -> list[str]:
    return sorted(df["Catégorie"].dropna().astype(str).unique().tolist())


def apply_filters(
    df: pd.DataFrame,
    include_mode: str,
    selected_categories: list[str],
    year_min: int | None,
    year_max: int | None,
) -> pd.DataFrame:
    """Filter dataframe by categories include/exclude and year range (if Année exists)."""
    filtered = df.copy()

    # Categories filter
    cats_all = available_categories(filtered)
    selected = [c for c in selected_categories if c in cats_all]

    if include_mode == "include":
        # If user selected categories -> keep only those
        if selected:
            filtered = filtered[filtered["Catégorie"].isin(selected)]
    else:
        # exclude mode
        if selected:
            filtered = filtered[~filtered["Catégorie"].isin(selected)]

    # Year filter (only if Année exists)
    if "Année" in filtered.columns:
        if year_min is not None:
            filtered = filtered[(filtered["Année"].isna()) | (filtered["Année"] >= year_min)]
        if year_max is not None:
            filtered = filtered[(filtered["Année"].isna()) | (filtered["Année"] <= year_max)]

    return filtered


def draw_selection(
    df: pd.DataFrame,
    mode: str,
    k_categories: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns (selection_df, meta)
    meta may include chosen_categories, winner_row, etc.
    """
    meta: dict = {}

    if df.empty:
        return pd.DataFrame(), meta

    cats = available_categories(df)
    if not cats:
        return pd.DataFrame(), meta

    # clamp k
    k = max(1, min(int(k_categories), len(cats)))

    if mode == "equilibre":
        # Randomly choose k categories then one movie per chosen category
        chosen_categories = pd.Series(cats).sample(n=k, replace=False).tolist()
        meta["chosen_categories"] = chosen_categories

        df_subset = df[df["Catégorie"].isin(chosen_categories)]
        selection = (
            df_subset.groupby("Catégorie", as_index=False)
            .sample(n=1)
            .reset_index(drop=True)
        )
        return selection, meta

    if mode == "aleatoire":
        # Random 3 movies (or less if df smaller)
        n = min(3, len(df))
        selection = df.sample(n=n).reset_index(drop=True)
        return selection, meta

    if mode == "roulette":
        winner = df.sample(n=1).iloc[0]
        meta["winner"] = winner.to_dict()
        selection = pd.DataFrame([winner]).reset_index(drop=True)
        return selection, meta

    # default fallback
    selection = df.sample(n=1).reset_index(drop=True)
    return selection, meta


def selection_to_text(selection: pd.DataFrame, meta: dict) -> str:
    """Create a nice text summary to copy/paste."""
    lines = []
    if "chosen_categories" in meta:
        lines.append("Catégories tirées : " + ", ".join(meta["chosen_categories"]))
        lines.append("")

    if selection is None or selection.empty:
        return "Aucune sélection."

    # Columns
    has_year = "Année" in selection.columns

    for _, row in selection.iterrows():
        cat = str(row.get("Catégorie", ""))
        title = str(row.get("Titre", ""))
        if has_year and pd.notna(row.get("Année", None)):
            year = int(row["Année"])
            lines.append(f"• [{cat}] {title} ({year})")
        else:
            lines.append(f"• [{cat}] {title}")

    return "\n".join(lines)


# --- GUI ---
class MovieSelectorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎬 Sélecteur de films")
        self.root.geometry("980x640")
        self.root.minsize(920, 600)

        # State
        self.df_all: pd.DataFrame | None = None
        self.df_filtered: pd.DataFrame | None = None

        self.mode_var = tk.StringVar(value="equilibre")
        self.mode_dropdown_var = tk.StringVar(value="Équilibré")
        self.include_mode_var = tk.StringVar(value="include")  # include/exclude
        self.k_var = tk.IntVar(value=3)

        self.year_min_var = tk.StringVar(value="")
        self.year_max_var = tk.StringVar(value="")

        self.history: list[dict] = []  # each item: {timestamp, mode, k, text, selection_df, meta}

        # Build UI
        self._build_layout()

        # Load data
        self._load_data()

    def _build_layout(self):
        # Notebook for tabs
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_main = ttk.Frame(self.nb)
        self.tab_history = ttk.Frame(self.nb)

        self.nb.add(self.tab_main, text="Sélection")
        self.nb.add(self.tab_history, text="Historique")

        # --- Main tab layout: left controls / right results ---
        self.tab_main.columnconfigure(0, weight=0)
        self.tab_main.columnconfigure(1, weight=1)
        self.tab_main.rowconfigure(0, weight=1)

        left = ttk.Frame(self.tab_main, padding=10)
        right = ttk.Frame(self.tab_main, padding=10)
        left.grid(row=0, column=0, sticky="nsw")
        right.grid(row=0, column=1, sticky="nsew")

        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # --- Controls (left) ---
        ttk.Label(left, text="Mode", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 6))

        # Quick buttons
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(0, 6))
        ttk.Button(btns, text="⚖️ Équilibré", command=lambda: self.set_mode("equilibre")).grid(row=0, column=0, padx=2)
        ttk.Button(btns, text="🎲 Aléatoire", command=lambda: self.set_mode("aleatoire")).grid(row=0, column=1, padx=2)
        ttk.Button(btns, text="🎯 Roulette", command=lambda: self.set_mode("roulette")).grid(row=0, column=2, padx=2)

        # Radiobuttons
        rb = ttk.Frame(left)
        rb.pack(fill="x", pady=(0, 6))
        ttk.Radiobutton(rb, text="Équilibré", value="equilibre", variable=self.mode_var, command=self._sync_mode_widgets).pack(anchor="w")
        ttk.Radiobutton(rb, text="Aléatoire (3 films)", value="aleatoire", variable=self.mode_var, command=self._sync_mode_widgets).pack(anchor="w")
        ttk.Radiobutton(rb, text="Roulette (1 gagnant)", value="roulette", variable=self.mode_var, command=self._sync_mode_widgets).pack(anchor="w")

        # Dropdown
        ttk.Label(left, text="Menu déroulant", font=("Arial", 10, "bold")).pack(anchor="w", pady=(8, 2))
        options = ["Équilibré", "Aléatoire (3 films)", "Roulette (1 gagnant)"]
        self.mode_menu = ttk.OptionMenu(left, self.mode_dropdown_var, options[0], *options, command=self._on_dropdown_change)
        self.mode_menu.pack(fill="x")

        ttk.Separator(left).pack(fill="x", pady=10)

        # K categories controls (only meaningful for équilibré)
        ttk.Label(left, text="Nombre de catégories (mode Équilibré)", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 4))

        krow = ttk.Frame(left)
        krow.pack(fill="x", pady=(0, 8))
        ttk.Label(krow, text="Spinbox:").grid(row=0, column=0, sticky="w")
        self.k_spin = ttk.Spinbox(krow, from_=1, to=99, textvariable=self.k_var, width=6, command=self._on_k_change)
        self.k_spin.grid(row=0, column=1, sticky="w", padx=(6, 0))

        ttk.Label(left, text="Slider:").pack(anchor="w")
        self.k_scale = ttk.Scale(left, from_=1, to=10, orient="horizontal", command=self._on_scale_move)
        self.k_scale.pack(fill="x", pady=(0, 6))

        ttk.Separator(left).pack(fill="x", pady=10)

        # Filters
        ttk.Label(left, text="Filtres", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 6))

        inc_exc = ttk.Frame(left)
        inc_exc.pack(fill="x", pady=(0, 6))
        ttk.Radiobutton(inc_exc, text="Inclure uniquement", value="include", variable=self.include_mode_var, command=self._refresh_filtered).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(inc_exc, text="Exclure", value="exclude", variable=self.include_mode_var, command=self._refresh_filtered).grid(row=1, column=0, sticky="w")

        ttk.Label(left, text="Catégories (Ctrl/Shift pour multi-sélection):").pack(anchor="w")
        self.cats_listbox = tk.Listbox(left, selectmode="extended", height=10)
        self.cats_listbox.pack(fill="both", expand=False, pady=(0, 6))
        self.cats_listbox.bind("<<ListboxSelect>>", lambda e: self._refresh_filtered())

        # Year filter widgets
        self.year_frame = ttk.LabelFrame(left, text="Année (si disponible)")
        self.year_frame.pack(fill="x", pady=(8, 6))
        yrow = ttk.Frame(self.year_frame)
        yrow.pack(fill="x", padx=8, pady=8)
        ttk.Label(yrow, text="Min:").grid(row=0, column=0, sticky="w")
        self.year_min_entry = ttk.Entry(yrow, textvariable=self.year_min_var, width=8)
        self.year_min_entry.grid(row=0, column=1, sticky="w", padx=(6, 14))
        ttk.Label(yrow, text="Max:").grid(row=0, column=2, sticky="w")
        self.year_max_entry = ttk.Entry(yrow, textvariable=self.year_max_var, width=8)
        self.year_max_entry.grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Button(self.year_frame, text="Appliquer", command=self._refresh_filtered).pack(anchor="e", padx=8, pady=(0, 8))

        ttk.Separator(left).pack(fill="x", pady=10)

        # Actions
        actions = ttk.Frame(left)
        actions.pack(fill="x")
        ttk.Button(actions, text="🎬 Tirer", command=self.run_draw).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(actions, text="🔄 Recharger la feuille", command=self._load_data).grid(row=0, column=1, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        ttk.Button(left, text="📋 Copier la sélection", command=self.copy_selection).pack(fill="x", pady=(8, 0))

        # --- Results (right) ---
        header = ttk.Frame(right)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.status_lbl = ttk.Label(header, text="Chargement...", font=("Arial", 11))
        self.status_lbl.grid(row=0, column=0, sticky="w")

        # Toggle view
        view_row = ttk.Frame(right)
        view_row.grid(row=1, column=0, sticky="nsew")
        view_row.rowconfigure(1, weight=1)
        view_row.columnconfigure(0, weight=1)

        self.view_mode = tk.StringVar(value="table")
        view_switch = ttk.Frame(view_row)
        view_switch.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(view_switch, text="Tableau", value="table", variable=self.view_mode, command=self._update_view_visibility).pack(side="left")
        ttk.Radiobutton(view_switch, text="Liste simple", value="list", variable=self.view_mode, command=self._update_view_visibility).pack(side="left", padx=(12, 0))

        # Treeview (table)
        self.tree = ttk.Treeview(view_row, columns=("Catégorie", "Titre", "Année"), show="headings", height=12)
        self.tree.grid(row=1, column=0, sticky="nsew")

        self.tree.heading("Catégorie", text="Catégorie", command=lambda: self.sort_tree("Catégorie"))
        self.tree.heading("Titre", text="Titre", command=lambda: self.sort_tree("Titre"))
        self.tree.heading("Année", text="Année", command=lambda: self.sort_tree("Année"))

        self.tree.column("Catégorie", width=180, anchor="w")
        self.tree.column("Titre", width=520, anchor="w")
        self.tree.column("Année", width=80, anchor="center")

        # Scrollbar for tree
        tree_scroll = ttk.Scrollbar(view_row, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.grid(row=1, column=1, sticky="ns")

        # List view
        self.listbox = tk.Listbox(view_row, height=12)
        self.listbox.grid(row=1, column=0, sticky="nsew")
        self.listbox_scroll = ttk.Scrollbar(view_row, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=self.listbox_scroll.set)
        self.listbox_scroll.grid(row=1, column=1, sticky="ns")

        # default to table visible
        self._update_view_visibility()

        # --- History tab ---
        self.tab_history.columnconfigure(0, weight=1)
        self.tab_history.rowconfigure(1, weight=1)

        ttk.Label(self.tab_history, text="Historique des tirages", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        self.hist_tree = ttk.Treeview(self.tab_history, columns=("Date", "Mode", "Détails"), show="headings")
        self.hist_tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        self.hist_tree.heading("Date", text="Date")
        self.hist_tree.heading("Mode", text="Mode")
        self.hist_tree.heading("Détails", text="Détails")

        self.hist_tree.column("Date", width=180, anchor="w")
        self.hist_tree.column("Mode", width=140, anchor="w")
        self.hist_tree.column("Détails", width=640, anchor="w")

        self.hist_tree.bind("<<TreeviewSelect>>", self._on_history_select)

        self.hist_details = tk.Text(self.tab_history, height=10, wrap="word")
        self.hist_details.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.hist_details.configure(state="disabled")

    # --- Mode syncing ---
    def set_mode(self, mode: str):
        self.mode_var.set(mode)
        self._sync_mode_widgets()

    def _on_dropdown_change(self, _value):
        mapping = {
            "Équilibré": "equilibre",
            "Aléatoire (3 films)": "aleatoire",
            "Roulette (1 gagnant)": "roulette",
        }
        self.mode_var.set(mapping.get(self.mode_dropdown_var.get(), "equilibre"))
        self._sync_mode_widgets()

    def _sync_mode_widgets(self):
        # sync dropdown based on radio/buttons
        inv = {
            "equilibre": "Équilibré",
            "aleatoire": "Aléatoire (3 films)",
            "roulette": "Roulette (1 gagnant)",
        }
        self.mode_dropdown_var.set(inv.get(self.mode_var.get(), "Équilibré"))
        self._update_k_controls_state()

    def _update_k_controls_state(self):
        is_equilibre = self.mode_var.get() == "equilibre"
        state = "normal" if is_equilibre else "disabled"
        try:
            self.k_spin.configure(state=state)
        except tk.TclError:
            pass
        # ttk.Scale doesn't have "disabled" in the same way, but we can set it
        self.k_scale.state(["!disabled"] if is_equilibre else ["disabled"])

    # --- Data loading / filtering ---
    def _load_data(self):
        self.status_lbl.config(text="📡 Connexion à Google Sheets...")
        self.root.update_idletasks()

        try:
            self.df_all = fetch_data()
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self.status_lbl.config(text="❌ Erreur de chargement.")
            self.df_all = None
            return

        self.status_lbl.config(text=f"✅ {len(self.df_all)} films 'Pas encore vu' chargés.")
        self._populate_categories()
        self._refresh_filtered()

    def _populate_categories(self):
        self.cats_listbox.delete(0, tk.END)
        if self.df_all is None:
            return
        for cat in available_categories(self.df_all):
            self.cats_listbox.insert(tk.END, cat)

    def _get_selected_categories(self) -> list[str]:
        # from listbox selection indices
        selected = []
        for i in self.cats_listbox.curselection():
            selected.append(self.cats_listbox.get(i))
        return selected

    def _parse_year(self, s: str) -> int | None:
        s = (s or "").strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None

    def _refresh_filtered(self):
        if self.df_all is None:
            return

        year_min = self._parse_year(self.year_min_var.get())
        year_max = self._parse_year(self.year_max_var.get())
        if (self.year_min_var.get().strip() and year_min is None) or (self.year_max_var.get().strip() and year_max is None):
            messagebox.showwarning("Année", "Filtre année invalide (entiers attendus).")
            return

        self.df_filtered = apply_filters(
            self.df_all,
            include_mode=self.include_mode_var.get(),
            selected_categories=self._get_selected_categories(),
            year_min=year_min,
            year_max=year_max,
        )

        # Update available categories for k max (based on filtered df)
        cats = available_categories(self.df_filtered)
        max_k = max(1, len(cats))
        self._update_k_range(max_k)

        self.status_lbl.config(text=f"🎬 Films disponibles après filtres : {len(self.df_filtered)} | Catégories : {len(cats)}")

    def _update_k_range(self, max_k: int):
        # Update spinbox and scale bounds
        current = int(self.k_var.get())
        if current > max_k:
            self.k_var.set(max_k)
        if current < 1:
            self.k_var.set(1)

        try:
            self.k_spin.configure(from_=1, to=max_k)
        except tk.TclError:
            pass

        self.k_scale.configure(from_=1, to=max_k)
        # Keep scale in sync
        self.k_scale.set(self.k_var.get())

    def _on_k_change(self):
        # spinbox changed -> update scale
        try:
            v = int(self.k_var.get())
            self.k_scale.set(v)
        except Exception:
            pass

    def _on_scale_move(self, value):
        # scale moved -> update spinbox int var
        try:
            self.k_var.set(int(float(value)))
        except Exception:
            pass

    # --- Results display ---
    def _update_view_visibility(self):
        if self.view_mode.get() == "table":
            self.listbox.grid_remove()
            self.listbox_scroll.grid_remove()
            self.tree.grid()
            # scrollbar already in same cell; keep
            self.tree.yview_moveto(0)
        else:
            self.tree.grid_remove()
            # its scrollbar stays; hide it and show listbox scroll
            self.listbox.grid()
            self.listbox_scroll.grid()
            self.listbox.yview_moveto(0)

    def clear_results(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.listbox.delete(0, tk.END)

    def show_selection(self, selection: pd.DataFrame, meta: dict):
        self.clear_results()

        if selection is None or selection.empty:
            self.listbox.insert(tk.END, "Aucune sélection.")
            return

        # Prepare for tree
        has_year = "Année" in selection.columns

        # Alternate row highlighting + roulette highlight
        winner_title = None
        if "winner" in meta and meta["winner"]:
            winner_title = str(meta["winner"].get("Titre", ""))

        for idx, row in selection.iterrows():
            cat = str(row.get("Catégorie", ""))
            title = str(row.get("Titre", ""))
            year = row.get("Année", None) if has_year else None

            year_str = ""
            if has_year and pd.notna(year):
                try:
                    year_str = str(int(year))
                except Exception:
                    year_str = str(year)

            tags = []
            # zebra stripes
            tags.append("odd" if idx % 2 else "even")
            # roulette winner emphasis
            if winner_title and title == winner_title:
                tags.append("winner")

            self.tree.insert("", tk.END, values=(cat, title, year_str), tags=tuple(tags))

            # list view text
            if year_str:
                self.listbox.insert(tk.END, f"• [{cat}] {title} ({year_str})")
            else:
                self.listbox.insert(tk.END, f"• [{cat}] {title}")

        # Configure tag styles (basic highlight)
        self.tree.tag_configure("even", background="#FFFFFF")
        self.tree.tag_configure("odd", background="#F6F6F6")
        self.tree.tag_configure("winner", background="#FFF3B0")  # light highlight

        # Extra info in status (chosen categories)
        if "chosen_categories" in meta:
            self.status_lbl.config(
                text=f"✅ Tirage OK | Catégories tirées : {', '.join(meta['chosen_categories'])}"
            )
        elif winner_title:
            self.status_lbl.config(text=f"🏆 Gagnant : {winner_title}")

    def sort_tree(self, col: str):
        """Sort treeview by column (toggle asc/desc)."""
        data = []
        for iid in self.tree.get_children():
            values = self.tree.item(iid, "values")
            data.append((iid, values))

        col_idx = {"Catégorie": 0, "Titre": 1, "Année": 2}[col]

        # toggle sort direction
        current = getattr(self, "_sort_state", {})
        desc = current.get(col, False)
        current[col] = not desc
        self._sort_state = current

        def key_fn(item):
            v = item[1][col_idx]
            # numeric sort for year
            if col == "Année":
                try:
                    return int(v)
                except Exception:
                    return -10**9
            return str(v).lower()

        data.sort(key=key_fn, reverse=desc)

        for i, (iid, _) in enumerate(data):
            self.tree.move(iid, "", i)

    # --- Actions ---
    def run_draw(self):
        if self.df_filtered is None:
            messagebox.showwarning("Données", "Aucune donnée disponible.")
            return

        if self.df_filtered.empty:
            messagebox.showwarning("Filtres", "Aucun film ne correspond aux filtres.")
            return

        mode = self.mode_var.get()
        k = int(self.k_var.get())

        selection, meta = draw_selection(self.df_filtered, mode=mode, k_categories=k)
        self.show_selection(selection, meta)

        # Save history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode_label = {"equilibre": "Équilibré", "aleatoire": "Aléatoire", "roulette": "Roulette"}.get(mode, mode)
        text = selection_to_text(selection, meta)

        item = {
            "timestamp": timestamp,
            "mode": mode_label,
            "k": k,
            "text": text,
            "meta": meta,
        }
        self.history.insert(0, item)  # newest first
        self._refresh_history_view(item)

    def copy_selection(self):
        # copy latest selection from status/results by reconstructing from view
        if not self.history:
            messagebox.showinfo("Copier", "Aucun tirage à copier pour le moment.")
            return

        text = self.history[0]["text"]
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()  # ensures clipboard kept after app closes
        messagebox.showinfo("Copier", "✅ Sélection copiée dans le presse-papiers !")

    # --- History tab ---
    def _refresh_history_view(self, newest_item: dict):
        # Insert top row
        details_preview = newest_item["text"].splitlines()[0] if newest_item["text"] else ""
        self.hist_tree.insert("", 0, values=(newest_item["timestamp"], newest_item["mode"], details_preview))

        # Keep history tree reasonable
        max_rows = 200
        children = self.hist_tree.get_children()
        if len(children) > max_rows:
            for iid in children[max_rows:]:
                self.hist_tree.delete(iid)

        # Update details box to newest
        self._set_history_details(newest_item["text"])

    def _on_history_select(self, _event):
        sel = self.hist_tree.selection()
        if not sel:
            return
        # Find index by row position (since we insert at top in same order)
        iid = sel[0]
        row_index = self.hist_tree.index(iid)
        if 0 <= row_index < len(self.history):
            self._set_history_details(self.history[row_index]["text"])

    def _set_history_details(self, text: str):
        self.hist_details.configure(state="normal")
        self.hist_details.delete("1.0", tk.END)
        self.hist_details.insert(tk.END, text)
        self.hist_details.configure(state="disabled")


def main():
    root = tk.Tk()
    # Use a nicer theme if available
    try:
        style = ttk.Style()
        if os.name == "nt":
            style.theme_use("vista")
        else:
            # fallback to default / clam
            if "clam" in style.theme_names():
                style.theme_use("clam")
    except Exception:
        pass

    app = MovieSelectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
