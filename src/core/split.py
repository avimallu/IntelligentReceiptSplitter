import polars as pl


class IncompleteSplitError(Exception):
    def __init__(
        self,
        message,
    ):
        super().__init__(message)


class SplitCalculator:
    """
    A simple, but long class to calculate splits for a provided receipt.

    Args:
        item_names: Names of the items being split.
        item_people: A list of people for each item who are splitting its cost.
        item_amounts: Amounts of the items being split.
        total: The total amount in the receipt
        tip: The tip in the receipt
        tax: The tax in the receipt
        people_list: The total number of people splitting the receipt.
        tip_split_proportionally: Indicator for whether the tip is split proportional to pre-tax/tip cost.
        tax_split_proportionally: Indicator for whether the tax is split proportional to pre-tax/tip cost.
        cashback_discount: The total will be reduced by this percentage value.
        return_detailed_table: Indicator to return full calculation table or a simplified one.
    """

    def __init__(
        self,
        item_names: list[str],
        item_people: list[list[str]],
        item_amounts: list[float],
        receipt_total: float,
        receipt_tip: float,
        receipt_tax: float,
        people_list: list[str],
        tip_split_proportionally: bool,
        tax_split_proportionally: bool,
        cashback_discount: float,
        return_detailed_table: bool = False,
    ):
        self.item_names = item_names
        self.item_people = item_people
        self.item_amounts = item_amounts
        self.receipt_total = receipt_total
        self.receipt_tip = receipt_tip
        self.receipt_tax = receipt_tax
        self.people_list = people_list
        self.tip_split_proportionally = tip_split_proportionally
        self.tax_split_proportionally = tax_split_proportionally
        self.cashback_discount = cashback_discount
        self.return_detailed_table = return_detailed_table

        self.subtotal = self.receipt_total - self.receipt_tip - self.receipt_tax
        self.split_tips: float | None = None
        self.split_taxes: float | None = None

    def validate_splits(self):
        split_count = 0
        unsplit_names = []
        for name, split in zip(self.item_names, self.item_people):
            if len(split) > 0:
                split_count += 1
            else:
                unsplit_names.append(name)
        if split_count != len(self.item_people):
            raise IncompleteSplitError(
                f"The following items have not been assigned splits: {','.join(unsplit_names)}"
            )

    def distribute_amount(self, amount: float, split_subtotals: list[float]):
        """
        Distribute `amount` equally, or distribute it proportionally, among
        all the people involved in the split.
        """
        return [
            x / self.subtotal * amount
            if self.tax_split_proportionally
            else amount / len(self.people_list)
            for x in split_subtotals
        ]

    def subtract_cashback(
        self, split_totals: list[float]
    ) -> tuple[list[float], list[float]]:
        split_cashback = [-x * cashback_discount for x in split_totals]
        split_totals_minus_cashback = [
            x * (1 - cashback_discount) for x in split_totals
        ]
        return split_cashback, split_totals_minus_cashback

    def forward(self):
        split_arrays: list[list[float]] = []
        for split in self.item_people:
            split_array = [
                1 / len(split) if x in split else 0.0 for x in self.people_list
            ]
            split_arrays.append(split_array)
        split_amounts: list[list[float]] = []
        for split_array, amount in zip(split_arrays, self.item_amounts):
            split_amount = [amount * split for split in split_array]
            split_amounts.append(split_amount)

        split_subtotals = [sum(x) for x in zip(*split_amounts)]
        split_tips = self.distribute_amount(self.receipt_tip, split_subtotals)
        split_taxes = self.distribute_amount(self.receipt_tax, split_subtotals)

        split_totals = [
            split_subtotal + split_tip + split_tax
            for split_subtotal, split_tip, split_tax in zip(
                split_subtotals, split_tips, split_taxes
            )
        ]
        split_cashback, split_totals_minus_cashback = self.subtract_cashback(
            split_totals
        )


def calculate_splits(
    item_names: list[str],
    item_people: list[list[str]],
    item_amounts: list[float],
    total: float,
    tip: float,
    tax: float,
    people_list: list[str],
    tip_split_proportionally: bool,
    tax_split_proportionally: bool,
    cashback_discount: float,
    return_detailed_table: bool = False,
) -> pl.DataFrame:
    """
    A simple, but long function to calculate splits for a provided receipt.

    Args:
        item_names: Names of the items being split.
        item_people: A list of people for each item who are splitting its cost.
        item_amounts: Amounts of the items being split.
        total: The total amount in the receipt
        tip: The tip in the receipt
        tax: The tax in the receipt
        people_list: The total number of people splitting the receipt.
        tip_split_proportionally: Indicator for whether the tip is split proportional to pre-tax/tip cost.
        tax_split_proportionally: Indicator for whether the tax is split proportional to pre-tax/tip cost.
        cashback_discount: The total will be reduced by this percentage value.
        return_detailed_table: Indicator to return full calculation table or a simplified one.

    Returns:
        A DataFrame form of the provided values along with their calculated splits or a simplified version.
    """
    split_count = 0
    unsplit_names = []
    checkbox_count = len(item_people)
    for name, split in zip(item_names, item_people):
        if len(split) > 0:
            split_count += 1
        else:
            unsplit_names.append(name)
    if split_count != checkbox_count:
        raise IncompleteSplitError(
            f"The following items have not been assigned splits: {','.join(unsplit_names)}"
        )
        return None
    else:
        # Deliberately avoiding going the numpy route here since the data is very small anyway.
        split_arrays: list[list[float]] = []
        for split in item_people:
            split_array = [1 / len(split) if x in split else 0.0 for x in people_list]
            split_arrays.append(split_array)
        split_amounts: list[list[float]] = []
        for split_array, amount in zip(split_arrays, item_amounts):
            split_amount = [amount * split for split in split_array]
            split_amounts.append(split_amount)

        split_subtotals = [sum(x) for x in zip(*split_amounts)]
        subtotal = total - tip - tax
        split_tips = [
            x / subtotal * tip if tip_split_proportionally else tax / len(people_list)
            for x in split_subtotals
        ]
        split_taxes = [
            x / subtotal * tax if tax_split_proportionally else tax / len(people_list)
            for x in split_subtotals
        ]
        split_totals_pre_cashback = [
            split_subtotal + split_tip + split_tax
            for split_subtotal, split_tip, split_tax in zip(
                split_subtotals, split_tips, split_taxes
            )
        ]
        split_cashback = [-x * cashback_discount for x in split_totals_pre_cashback]
        split_totals_post_cashback = [
            x * (1 - cashback_discount) for x in split_totals_pre_cashback
        ]
        first_col_names = list(item_names) + [
            "Subtotal",
            "Tip",
            "Tax",
            "Cashback",
            "Total",
        ]
        splits = split_amounts + [
            split_subtotals,
            split_tips,
            split_taxes,
            split_cashback,
            split_totals_post_cashback,
        ]
        horizontal_totals = list(item_amounts) + [
            subtotal,
            tip,
            tax,
            sum(split_cashback),
            sum(split_totals_post_cashback),
        ]
        full_calculation_df = (
            pl.DataFrame(
                {
                    "Item": first_col_names,
                    "splits": splits,
                    "Total": horizontal_totals,
                },
                schema={
                    "Item": pl.String,
                    "splits": pl.List(pl.Float64),
                    "Total": pl.Float64,
                },
            )
            .with_columns(pl.col("splits").list.to_struct(fields=people_list))
            .unnest("splits")
            .with_columns(pl.col(pl.Float64).round(2))
        )
        if return_detailed_table:
            return full_calculation_df
        else:
            simple_calculation = (
                full_calculation_df.filter(pl.col("Item").eq("Total"))
                .select(pl.exclude("Total"))
                .transpose(
                    include_header=True, header_name="Person", column_names=["Split"]
                )
                .filter(pl.col("Person").ne("Item"))
            )
            return simple_calculation


if __name__ == "__main__":
    # Example usage
    item_names = ["Item 1", "Item 2", "Item 3"]
    item_people = [["Alice", "Bob"], ["Alice"], ["Bob", "Charlie"]]
    item_amounts = [10.0, 20.0, 30.0]
    total = 70.0
    tip = 6.0
    tax = 4.0
    people_list = ["Alice", "Bob", "Charlie"]
    tip_split_proportionally = True
    tax_split_proportionally = True
    cashback_discount = 0.03

    result_df = calculate_splits(
        item_names,
        item_people,
        item_amounts,
        total,
        tip,
        tax,
        people_list,
        tip_split_proportionally,
        tax_split_proportionally,
        cashback_discount,
        return_detailed_table=True,
    )
    print(result_df)
