"""Choose a CSV category in a popup, then review a random sample.

Run: python review_statements.py
Use --seed 42 for a reproducible sample. The original CSV is never modified.
Decisions are saved after every click; a filtered copy is saved on completion.
Closing early preserves decisions but does not produce a filtered copy.
"""

import argparse
import csv
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText


DEFAULT_CSV = Path(
    "/Users/saeedalahmari/Documents/LLM_ensemble_USF/code/LLMFrontEnd/"
    "mbzuai_extended_with_correct.csv"
)


def load_sample(csv_path, sample_size=100, skip=0, seed=None, category=None):
    """Skip within each category in file order, then sample without replacement."""
    if sample_size < 1 or skip < 0:
        raise ValueError("Sample size must be positive and skip must be nonnegative.")
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        if not fields or not {"question", "category"}.issubset(fields):
            raise ValueError("The CSV must contain 'question' and 'category' columns.")
        rows = list(reader)
    if not rows:
        raise ValueError("The CSV contains no statements.")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if not row["category"] or not row["category"].strip():
            raise ValueError(f"CSV row {index + 2} has no category.")
        if not row["question"] or not row["question"].strip():
            raise ValueError(f"CSV row {index + 2} has no statement.")
        groups[row["category"]].append(index)
    rng = random.Random(seed)
    selected = []
    if category is not None and category not in groups:
        raise ValueError(f"Unknown category: {category!r}")
    for category_name, indices in groups.items():
        if category is not None and category_name != category:
            continue
        eligible = indices[skip:]
        if len(eligible) < sample_size:
            raise ValueError(
                f"Category {category_name!r} has only {len(eligible)} statements after "
                f"skipping {skip}; cannot sample {sample_size}."
            )
        selected.extend(rng.sample(eligible, sample_size))
    return fields, rows, selected


def choose_category(root, csv_path, sample_size, skip, seed):
    """Return the selected sample, or None if the category popup is closed."""
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"question", "category"}.issubset(reader.fieldnames):
            raise ValueError("The CSV must contain 'question' and 'category' columns.")
        counts = defaultdict(int)
        for row in reader:
            if row["category"] and row["category"].strip():
                counts[row["category"]] += 1
    if not counts:
        raise ValueError("The CSV contains no categories.")

    result = None
    dialog = tk.Toplevel(root)
    dialog.title("Choose a category to review")
    dialog.geometry("520x260")
    dialog.resizable(False, False)
    tk.Label(dialog, text="Choose the category to verify:",
             font=("Arial", 14)).pack(anchor="w", padx=24, pady=(24, 12))
    categories = list(counts)
    selector = ttk.Combobox(dialog, values=categories + ["All categories"],
                            state="readonly", width=42)
    selector.current(0)
    selector.pack(padx=24, fill="x")
    details = tk.StringVar()

    def update_details(event=None):
        choice = selector.current()
        count = max(0, counts[categories[choice]] - skip) if choice < len(categories) else None
        details.set(
            f"Skip the first {skip} statements in each selected category.\n"
            + (f"Randomly review {sample_size} of {count} remaining statements."
               if count is not None else f"Randomly review {sample_size} statements per category.")
        )

    def start_review():
        nonlocal result
        choice = selector.current()
        category = categories[choice] if choice < len(categories) else None
        try:
            result = load_sample(csv_path, sample_size, skip, seed, category)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot review this category", str(exc), parent=dialog)
            return
        dialog.destroy()

    selector.bind("<<ComboboxSelected>>", update_details)
    tk.Label(dialog, textvariable=details, justify="left").pack(anchor="w", padx=24, pady=16)
    tk.Button(dialog, text="Start review", width=18, command=start_review).pack(pady=8)
    update_details()
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    root.wait_window(dialog)
    return result


def write_csv(path, fields, rows):
    """Replace an output atomically so interrupted writes do not truncate it."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class StatementReviewer:
    def __init__(self, root, fields, rows, selected, decisions_path, filtered_path):
        self.root = root
        self.fields = fields
        self.rows = rows
        self.selected = selected
        self.decisions_path = decisions_path
        self.filtered_path = filtered_path
        self.decisions = {}
        self.position = 0
        root.title("Statement review — Keep / Remove")
        root.geometry("850x500")
        root.minsize(600, 350)
        self.status = tk.StringVar()
        tk.Label(root, textvariable=self.status, font=("Arial", 14),
                 wraplength=750, justify="left").pack(anchor="w", padx=24, pady=20)
        self.statement = ScrolledText(root, wrap="word", font=("Arial", 18),
                                      height=8, padx=15, pady=15)
        self.statement.pack(fill="both", expand=True, padx=24)
        buttons = tk.Frame(root)
        buttons.pack(pady=20)
        self.keep = tk.Button(buttons, text="Keep", width=16, height=2,
                              command=lambda: self.decide("keep"))
        self.keep.pack(side="left", padx=12)
        self.remove = tk.Button(buttons, text="Remove", width=16, height=2,
                                command=lambda: self.decide("remove"))
        self.remove.pack(side="left", padx=12)
        tk.Label(root, text="Each choice is saved immediately. Original CSV is unchanged.").pack(pady=(0, 12))
        self.show_statement()

    def show_statement(self):
        index = self.selected[self.position]
        row = self.rows[index]
        self.status.set(
            f"Statement {self.position + 1} of {len(self.selected)}\n"
            f"Category: {row['category']}    |    CSV row: {index + 2}"
        )
        self.statement.configure(state="normal")
        self.statement.delete("1.0", "end")
        self.statement.insert("1.0", row["question"])
        self.statement.configure(state="disabled")

    def save_decisions(self):
        # Include the whole sample so an early exit also records pending items.
        write_csv(self.decisions_path,
                  ["source_csv_row", "decision", *self.fields],
                  ({**self.rows[index], "source_csv_row": index + 2,
                    "decision": self.decisions.get(index, "pending")}
                   for index in self.selected))

    def decide(self, decision):
        index = self.selected[self.position]
        self.decisions[index] = decision
        try:
            self.save_decisions()
        except OSError as exc:
            self.decisions.pop(index)
            messagebox.showerror("Could not save decision", str(exc), parent=self.root)
            return
        if self.position + 1 == len(self.selected):
            self.finish()
        else:
            self.position += 1
            self.show_statement()

    def finish(self):
        try:
            write_csv(self.filtered_path, self.fields,
                      (row for index, row in enumerate(self.rows)
                       if self.decisions.get(index) != "remove"))
        except OSError as exc:
            messagebox.showerror("Could not save filtered CSV",
                                 f"{exc}\n\nDecisions are saved. Click again to retry.",
                                 parent=self.root)
            return
        removed = sum(value == "remove" for value in self.decisions.values())
        messagebox.showinfo(
            "Review complete",
            f"Reviewed: {len(self.selected)}\nRemoved: {removed}\n"
            f"Rows in filtered CSV: {len(self.rows) - removed}\n\n"
            f"Filtered CSV:\n{self.filtered_path}\n\n"
            f"Decisions:\n{self.decisions_path}", parent=self.root,
        )
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-file", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--sample-size", type=int, default=20, help="Per category (default: 20).")
    parser.add_argument("--skip", type=int, default=10, help="First rows to skip per category (default: 10).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, help="Defaults to the input CSV's directory.")
    args = parser.parse_args()
    try:
        source = args.csv_file.expanduser().resolve()
        if args.sample_size < 1 or args.skip < 0:
            raise ValueError("Sample size must be positive and skip must be nonnegative.")
        root = tk.Tk()
        root.withdraw()
        sample = choose_category(root, source, args.sample_size, args.skip, args.seed)
        if sample is None:
            root.destroy()
            return
        fields, rows, selected = sample
        output_dir = (args.output_dir or source.parent).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{source.stem}_review_{datetime.now():%Y%m%d_%H%M%S_%f}"
        decisions_path = output_dir / f"{prefix}_decisions.csv"
        filtered_path = output_dir / f"{prefix}_filtered.csv"
        app = StatementReviewer(root, fields, rows, selected, decisions_path, filtered_path)
        app.save_decisions()
        root.deiconify()
    except (OSError, ValueError, tk.TclError) as exc:
        parser.exit(1, f"Cannot start review: {exc}\n")
    print(f"Reviewing {len(selected)} statements. Decisions: {decisions_path}")
    print(f"Filtered CSV will be saved on completion: {filtered_path}")
    root.mainloop()


if __name__ == "__main__":
    main()
