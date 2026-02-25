# -*- coding: utf-8 -*-
"""
Movie Selector - GUI (Tkinter)
Author: pineaulo (Enhanced by AI)
"""

import os
import re
import requests
import pandas as pd
from io import BytesIO
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox


# --- Configuration ---
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ-I3jNSGm4lQ4woB_RduhnQ-Y_23h-okao2BBRxGe4ESRfOatDNtl4ibZcN8HETMPDmvN_FuCzeKrC/pub?gid=0&single=true&output=csv"

YEAR_COL = "Année(s)"
YEAR_START_COL = "_YearStart"
YEAR_END_COL = "_YearEnd"

# --- Helpers ---
_YEAR_RANGE_RE = re.compile(r"^\s*(\d{4})\s*[-–—]\s*(\d{4})\s*$")
_YEAR_SINGLE_RE = re.compile(r"^\s*(\d{4})\s*$")


def parse_years_cell(cell) -> tuple[pd._libs.missing.NAType | int, pd._libs.missing.NAType | int, str]:
    """
    Returns (start_year, end_year, display_str).
    Accepts:
      - "1991"
      - "1991-2002" (also en-dash/em-dash)
    If invalid/empty -> (NA, NA, "")
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return (pd.NA, pd.NA, "")

    s = str(cell).strip()
    if not s:
        return (pd.NA, pd.NA, "")

    m = _YEAR_SINGLE_RE.match(s)
    if m:
        y = int(m.group(1))
        return (y, y, str(y))

    m = _YEAR_RANGE_RE.match(s)
    if m:
        y1 = int(m.group(1))
        y2 = int(m.group(2))
        # normalize if inverted
        start, end = (y1, y2) if y1 <= y2 else (y2, y1)
        return (start, end, f"{start}-{end}")

    # unknown format
    return (pd.NA, pd.NA, s)


def available_categories(df: pd.DataFrame) -> list[str]:
    return sorted(df["Catégorie"].dropna().astype(str).unique().tolist())


def available_titles(df: pd.DataFrame) -> list[str]:
    # stable + user-friendly ordering
    titles = df["Titre"].dropna().astype(str).str.strip()
    return sorted(titles.unique().tolist(), key=lambda x: x.lower())


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

    # Optional year column (Année(s)) -> parse start/end
    if YEAR_COL in df.columns:
        parsed = df[YEAR_COL].apply(parse_years_cell)
        df[YEAR_START_COL] = parsed.apply(lambda t: t[0])
        df[YEAR_END_COL] = parsed.apply(lambda t: t[1])
        # normalized display (keep original column name)
        df[YEAR_COL] = parsed.apply(lambda t: t[2])
    else:
        df[YEAR_START_COL] = pd.NA
        df[YEAR_END_COL] = pd.NA

    df_unseen = df[df["Statut"] == "Pas encore vu"].copy()
    if df_unseen.empty:
        raise RuntimeError("Aucun film 'Pas encore vu' trouvé.")

    return df_unseen


def apply_filters(
    df: pd.DataFrame,
    include_mode: str,
    selected_categories: list[str],
    excluded_titles: list[str],
    year_min: int | None,
    year_max: int | None,
) -> pd.DataFrame:
    """
    Filter dataframe by:
      - categories include/exclude
      - titles exclusion
      - strict year range using YearStart/YearEnd if available
    """
    filtered = df.copy()

    # Categories filter
    cats_all = available_categories(filtered)
    selected = [c for c in selected_categories if c in cats_all]

    if include_mode == "include":
        if selected:
            filtered = filtered[filtered["Catégorie"].isin(selected)]
    else:
        if selected:
            filtered = filtered[~filtered["Catégorie"].isin(selected)]

    # Exclude titles
    if excluded_titles:
        excluded_set = set(str(t).strip() for t in excluded_titles if str(t).strip())
        if excluded_set:
            filtered = filtered[~filtered["Titre"].astype(str).str.strip().isin(excluded_set)]

    # Year filter (strict)
    # Keep NA years (same spirit as your original code: unknown year shouldn't block)
    if YEAR_START_COL in filtered.columns and YEAR_END_COL in filtered.columns:
        if year_min is not None:
            filtered = filtered[(filtered[YEAR_START_COL].isna()) | (filtered[YEAR_START_COL] >= year_min)]
        if year_max is not None:
            filtered = filtered[(filtered[YEAR_END_COL].isna()) | (filtered[YEAR_END_COL] <= year_max)]

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
        n = min(3, len(df))
        selection = df.sample(n=n).reset_index(drop=True)
        return selection, meta

    if mode == "roulette":
        winner = df.sample(n=1).iloc[0]
        meta["winner"] = winner.to_dict()
        selection = pd.DataFrame([winner]).reset_index(drop=True)
        return selection, meta

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

    has_year = YEAR_COL in selection.columns

    for _, row in selection.iterrows():
        cat = str(row.get("Catégorie", ""))
        title = str(row.get("Titre", ""))
        if has_year:
            y = str(row.get(YEAR_COL, "")).strip()
            if y:
                lines.append(f"• [{cat}] {title} ({y})")
            else:
                lines.append(f"• [{cat}] {title}")
        else:
            lines.append(f"• [{cat}] {title}")

    return "\n".join(lines)


def _year_sort_key_from_display(s: str) -> tuple[int, int]:
    """
    For sorting displayed YEAR_COL values in the Treeview.
    Returns (start, end). Unknown -> (-inf, -inf).
    """
    start, end, _ = parse_years_cell(s)
    if start is pd.NA or end is pd.NA:
        return (-10**9, -10**9)
    return (int(start), int(end))


# --- GUI ---
class MovieSelectorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎬 Sélecteur de films")
        self.root.geometry("1050x690")
        self.root.minsize(980, 620)

        # State
        self.df_all: pd.DataFrame | None = None
        self.df_filtered: pd.DataFrame | None = None

        self.mode_var = tk.StringVar(value="equilibre")
        self.mode_dropdown_var = tk.StringVar(value="Équilibré")
        self.include_mode_var = tk.StringVar(value="include")  # include/exclude
        self.k_var = tk.IntVar(value=3)

        self.year_min_var = tk.StringVar(value="")
        self.year_max_var = tk.StringVar(value="")

        self.history: list[dict] = []

        # Build UI
        self._build_layout()

        # Load data
        self._load_data()

    def _build_layout(self):
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_main = ttk.Frame(self.nb)
        self.tab_history = ttk.Frame(self.nb)

        self.nb.add(self.tab_main, text="Sélection")
        self.nb.add(self.tab_history, text="Historique")

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

        ttk.Label(left, text="Mode de tirage :", font=("Arial", 10, "bold")).pack(anchor="w", pady=(8, 2))
        options = ["Équilibré (1 film par catégorie)", "Aléatoire (3 films random)", "Roulette (1 film gagnant)"]
        self.mode_menu = ttk.OptionMenu(left, self.mode_dropdown_var, options[0], *options, command=self._on_dropdown_change)
        self.mode_menu.pack(fill="x")

        ttk.Separator(left).pack(fill="x", pady=10)

        # ----- Section Nombre de catégories -----
        self.k_section = ttk.Frame(left)

        ttk.Label(self.k_section, text="Nombre de catégories").pack(anchor="w")

        krow = ttk.Frame(self.k_section)
        krow.pack(fill="x", pady=(0, 6))

        self.k_spin = ttk.Spinbox(krow, from_=1, to=99, textvariable=self.k_var, width=6)
        self.k_spin.pack(side="left")

        self.k_section.pack(fill="x", pady=(0, 6))

        ttk.Separator(left).pack(fill="x", pady=10)

        # Filters
        ttk.Label(left, text="Filtres", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 6))

        inc_exc = ttk.Frame(left)
        inc_exc.pack(fill="x", pady=(0, 6))
        ttk.Radiobutton(inc_exc, text="Inclure uniquement", value="include", variable=self.include_mode_var, command=self._refresh_filtered).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(inc_exc, text="Exclure", value="exclude", variable=self.include_mode_var, command=self._refresh_filtered).grid(row=1, column=0, sticky="w")

        ttk.Label(left, text="Catégories (Ctrl/Shift pour multi-sélection):").pack(anchor="w")
        self.cats_listbox = tk.Listbox(left, selectmode="extended", height=9)
        self.cats_listbox.pack(fill="both", expand=False, pady=(0, 6))
        self.cats_listbox.bind("<<ListboxSelect>>", lambda e: self._refresh_filtered())

        # NEW: Excluded movies listbox
        ttk.Label(left, text="Films à exclure (Ctrl/Shift pour multi-sélection):").pack(anchor="w", pady=(6, 0))
        self.excluded_listbox = tk.Listbox(left, selectmode="extended", height=8)
        self.excluded_listbox.pack(fill="both", expand=False, pady=(0, 6))
        self.excluded_listbox.bind("<<ListboxSelect>>", lambda e: self._refresh_filtered())

        # Year filter widgets
        self.year_frame = ttk.LabelFrame(left, text="Année(s) (si disponible)")
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

        view_row = ttk.Frame(right)
        view_row.grid(row=1, column=0, sticky="nsew")
        view_row.rowconfigure(1, weight=1)
        view_row.columnconfigure(0, weight=1)

        # Treeview (table)
        self.tree = ttk.Treeview(view_row, columns=("Catégorie", "Titre", YEAR_COL), show="headings", height=12)
        self.tree.grid(row=1, column=0, sticky="nsew")

        self.tree.heading("Catégorie", text="Catégorie", command=lambda: self.sort_tree("Catégorie"))
        self.tree.heading("Titre", text="Titre", command=lambda: self.sort_tree("Titre"))
        self.tree.heading(YEAR_COL, text=YEAR_COL, command=lambda: self.sort_tree(YEAR_COL))

        self.tree.column("Catégorie", width=180, anchor="w")
        self.tree.column("Titre", width=560, anchor="w")
        self.tree.column(YEAR_COL, width=110, anchor="center")

        tree_scroll = ttk.Scrollbar(view_row, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.grid(row=1, column=1, sticky="ns")

        # --- History tab ---
        self.tab_history.columnconfigure(0, weight=1)
        self.tab_history.rowconfigure(1, weight=1)

        ttk.Label(self.tab_history, text="Historique des tirages",
                  font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w",
                                                   padx=10, pady=(10, 6))

        self.hist_tree = ttk.Treeview(self.tab_history,
                                      columns=("Date", "Mode", "Détails"), show="headings")
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

        self.k_var.trace_add("write", lambda *_: self._on_k_change())

    def _on_k_change(self):
        try:
            k = int(self.k_var.get())
        except Exception:
            self.k_var.set(1)
            return

        if k < 1:
            self.k_var.set(1)

        self._update_k_controls_state()

    def _on_dropdown_change(self, _value):
        mapping = {
            "Équilibré (1 film par catégorie)": "equilibre",
            "Aléatoire (3 films random)": "aleatoire",
            "Roulette (1 film gagnant)": "roulette",
        }
        self.mode_var.set(mapping.get(self.mode_dropdown_var.get(), "equilibre"))
        self._update_k_controls_state()

    def _update_k_controls_state(self):
        is_equilibre = self.mode_var.get() == "equilibre"

        if is_equilibre:
            if not self.k_section.winfo_ismapped():
                self.k_section.pack(fill="x", pady=(0, 6))
        else:
            if self.k_section.winfo_ismapped():
                self.k_section.pack_forget()

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
        self._populate_excluded_titles()
        self._refresh_filtered()

    def _populate_categories(self):
        self.cats_listbox.delete(0, tk.END)
        if self.df_all is None:
            return
        for cat in available_categories(self.df_all):
            self.cats_listbox.insert(tk.END, cat)

    def _populate_excluded_titles(self):
        self.excluded_listbox.delete(0, tk.END)
        if self.df_all is None:
            return
        for title in available_titles(self.df_all):
            self.excluded_listbox.insert(tk.END, title)

    def _get_selected_categories(self) -> list[str]:
        return [self.cats_listbox.get(i) for i in self.cats_listbox.curselection()]

    def _get_excluded_titles(self) -> list[str]:
        return [self.excluded_listbox.get(i) for i in self.excluded_listbox.curselection()]

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
            messagebox.showwarning("Année(s)", "Filtre année invalide (entiers attendus).")
            return

        self.df_filtered = apply_filters(
            self.df_all,
            include_mode=self.include_mode_var.get(),
            selected_categories=self._get_selected_categories(),
            excluded_titles=self._get_excluded_titles(),
            year_min=year_min,
            year_max=year_max,
        )

        cats = available_categories(self.df_filtered)
        max_k = max(1, len(cats))
        self._update_k_range(max_k)

        self.status_lbl.config(
            text=f"🎬 Films disponibles après filtres : {len(self.df_filtered)} | Catégories : {len(cats)}"
        )

    def _update_k_range(self, max_k: int):
        try:
            current = int(self.k_var.get())
        except Exception:
            current = 1
            self.k_var.set(1)

        if current > max_k:
            self.k_var.set(max_k)
        if current < 1:
            self.k_var.set(1)

        try:
            self.k_spin.configure(from_=1, to=max_k)
        except tk.TclError:
            pass

    def clear_results(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def show_selection(self, selection: pd.DataFrame, meta: dict):
        self.clear_results()

        if selection is None or selection.empty:
            return

        has_year = YEAR_COL in selection.columns

        winner_title = None
        if "winner" in meta and meta["winner"]:
            winner_title = str(meta["winner"].get("Titre", ""))

        for idx, row in selection.iterrows():
            cat = str(row.get("Catégorie", ""))
            title = str(row.get("Titre", ""))

            year_str = ""
            if has_year:
                year_str = str(row.get(YEAR_COL, "")).strip()

            tags = ["odd" if idx % 2 else "even"]
            if winner_title and title == winner_title:
                tags.append("winner")

            self.tree.insert("", tk.END, values=(cat, title, year_str), tags=tuple(tags))

        self.tree.tag_configure("even", background="#FFFFFF")
        self.tree.tag_configure("odd", background="#F6F6F6")
        self.tree.tag_configure("winner", background="#FFF3B0")

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

        col_idx = {"Catégorie": 0, "Titre": 1, YEAR_COL: 2}[col]

        current = getattr(self, "_sort_state", {})
        desc = current.get(col, False)
        current[col] = not desc
        self._sort_state = current

        def key_fn(item):
            v = item[1][col_idx]
            if col == YEAR_COL:
                start, end = _year_sort_key_from_display(str(v))
                return (start, end)
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
        self.history.insert(0, item)
        self._refresh_history_view(item)

    def copy_selection(self):
        if not self.history:
            messagebox.showinfo("Copier", "Aucun tirage à copier pour le moment.")
            return

        text = self.history[0]["text"]
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        messagebox.showinfo("Copier", "✅ Sélection copiée dans le presse-papiers !")

    # --- History tab ---
    def _refresh_history_view(self, newest_item: dict):
        details_preview = newest_item["text"].splitlines()[0] if newest_item["text"] else ""
        self.hist_tree.insert("", 0, values=(newest_item["timestamp"], newest_item["mode"], details_preview))

        max_rows = 200
        children = self.hist_tree.get_children()
        if len(children) > max_rows:
            for iid in children[max_rows:]:
                self.hist_tree.delete(iid)

        self._set_history_details(newest_item["text"])

    def _on_history_select(self, _event):
        sel = self.hist_tree.selection()
        if not sel:
            return
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
    try:
        style = ttk.Style()
        if os.name == "nt":
            style.theme_use("vista")
        else:
            if "clam" in style.theme_names():
                style.theme_use("clam")
    except Exception:
        pass

    app = MovieSelectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
