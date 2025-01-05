from datetime import datetime
from typing import Literal, TypedDict, TypeVar

import argparse as agp
import gradio as gr
import polars as pl
from PIL.Image import Image

from src.app.split_ai import ReceiptReader
from src.app.utils import css_code, head_html, spinner_html

ComponentType = TypeVar("ComponentType")

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
    return_detailed_table: bool = False
) -> gr.DataFrame:
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
        gr.Warning(
            f"Pending splits: {','.join(unsplit_names)}",
            title="Can't show splits yet",
        )
        return gr.DataFrame(pl.DataFrame(), visible=False)
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
        split_cashback = [
            - x * cashback_discount for x in split_totals_pre_cashback
        ]
        split_totals_post_cashback = [
            x * (1 - cashback_discount) for x in split_totals_pre_cashback
        ]
        first_col_names = list(item_names) + ["Subtotal", "Tip", "Tax", "Cashback", "Total"]
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
            return gr.DataFrame(full_calculation_df, visible=True)
        else:
            simple_calculation = (
                full_calculation_df
                .filter(pl.col("Item").eq("Total"))
                .select(pl.exclude("Total"))
                .transpose(include_header=True, header_name="Person", column_names=["Split"])
                .filter(pl.col("Person").ne("Item"))
            )
            return gr.DataFrame(simple_calculation, visible=True)


class Item(TypedDict):
    name: str
    amount: float


class ItemSplitter:
    def __init__(
        self,
        item: Item,
        people_list: list[str],
    ) -> None:
        self.people_list_state = people_list
        self.item = item
        self.no_interaction_kwargs = {"interactive": False, "min_width": 10}
        self.interaction_kwargs = self.no_interaction_kwargs | {
            "container": False,
            "interactive": True,
        }

    def name_textbox(self, item_name: str) -> gr.Textbox:
        return gr.Textbox(
            item_name, show_label=False, scale=8, **self.interaction_kwargs
        )

    def amount_number(self, item_amount: float) -> gr.Number:
        return gr.Number(
            value=item_amount, precision=2, scale=3, **self.interaction_kwargs
        )

    def split_status_button(
        self, choices: list[str] | None = None, status: Literal["⚠️", "🆗"] = "⚠️"
    ) -> gr.Button:
        if choices is not None:
            if len(choices) == 0:
                status = "⚠️"
            else:
                status = "🆗"
        else:
            choices = []
        variant: Literal["huggingface", "primary"] = (
            "huggingface" if (status == "⚠️") | (len(choices) == 0) else "primary"
        )
        return gr.Button(
            value=status, variant=variant, scale=1, **self.no_interaction_kwargs
        )

    def delete_item(self, item_list: list[Item]) -> list[Item]:
        item_list.remove(self.item)
        return item_list

    def delete_item_button(self) -> gr.Button:
        kwargs = self.no_interaction_kwargs | {"interactive": True}
        return gr.Button(value="❌", variant="stop", **kwargs)

    def people_list_checkbox(self, people_list: list[str]) -> gr.CheckboxGroup:
        return gr.CheckboxGroup(choices=people_list, **self.interaction_kwargs)

    def generate(self) -> tuple[gr.Textbox, gr.CheckboxGroup, gr.Number, gr.Button]:
        return self.generate_mobile()

    def generate_mobile(self):
        with gr.Row(variant="default", equal_height=True):
            item_name_textbox = self.name_textbox(self.item["name"])
            item_amount_number = self.amount_number(self.item["amount"])
            split_status_button = self.split_status_button(status="⚠️")
            delete_item_button = self.delete_item_button()
        people_list_checkbox = self.people_list_checkbox(self.people_list_state)
        people_list_checkbox.change(
            lambda x: self.split_status_button(choices=x),
            people_list_checkbox,
            split_status_button,
        )
        return (
            item_name_textbox,
            people_list_checkbox,
            item_amount_number,
            delete_item_button,
        )


class SplitAIApp:
    valid_split_variant: Literal["primary"] = "primary"
    invalid_split_variant: Literal["huggingface"] = "huggingface"

    def __init__(self, llm_model: str):
        self.receipt_reader = ReceiptReader(llm_model)
        self.demo = self.create_app()

    @staticmethod
    def prepare_calculate_splits_kwargs(num_records: int, *all_values) -> gr.DataFrame:
        """
        This method is necessary because only a list[gr.Component] or similar can be sent as
        `inputs` to an event listener. Therefore, it is unpacked here and prepared into a
        dictionary based on how it is sent by the event. This method is specifically for
        the `get_split_button.click` event listener.

        Args:
            num_records: The number of items present to split.
            *all_values: A list of components to forward to `calculate_splits`.

        Returns:
            gr.DataFrame
        """
        kwargs = {
            "item_names": all_values[:num_records],
            "item_people": all_values[num_records : num_records * 2],
            "item_amounts": all_values[num_records * 2 : num_records * 3],
        }
        additional_kwargs = {
            k: v
            for k, v in zip(
                [
                    "total",
                    "tip",
                    "tax",
                    "people_list",
                    "tip_split_proportionally",
                    "tax_split_proportionally",
                    "cashback_discount",
                    "return_detailed_table",
                ],
                tuple(all_values[num_records * 3 :]),
            )
        }
        additional_kwargs["cashback_discount"] /= 100
        kwargs.update(additional_kwargs)
        return calculate_splits(**kwargs)

    @staticmethod
    def update_component_attributes(component: ComponentType, **kwargs) -> ComponentType:
        """
        This requirement is in place because Gradio expects you to provide A NEW INSTANCE of
        the component that you want to update with its attributes changed. It seems like it
        doesn't replace the component, but updates it this way. Very weird behavior.

        Args:
            component: The gradio component to update attributes for.
            **kwargs: (attribute, value) pairs to update in child.

        Returns:
            A new instance of child's class with the updated attributes.
        """
        gradio_class = type(component)
        try:
            return gradio_class(**kwargs)
        except Exception as err:
            print(
                f"The Gradio component {gradio_class} does not have one of the provided attribute keys."
            )
            raise err

    @staticmethod
    def validate_people_list(people_textbox) -> tuple[gr.Image, list]:
        if "," in people_textbox and people_textbox[-1] != ",":
            people_list = [x.strip() for x in people_textbox.split(",")]
            return gr.Image(interactive=True), people_list
        else:
            gr.Warning("You need to enter a list of names separated by commas.")
            return gr.Image(interactive=False), []

    def create_app(self) -> gr.Blocks():
        # `head_html` required to prevent iOS from scaling the UI when clicking on a textbox.
        with gr.Blocks(
            css=css_code,
            head=head_html,
            theme="JohnSmith9982/small_and_pretty",
            fill_width=True,
        ) as split_app:
            with gr.Column():
                self.people_textbox = gr.Textbox(
                    placeholder="Split names with a comma",
                    label="Who all are splitting this receipt?",
                    lines=1,
                    autofocus=True,
                    submit_btn="Submit",
                )
                self.people_list = gr.State([])
                self.image_uploader = gr.Image(
                    show_label=False, scale=1, type="pil", interactive=False
                )
                self.people_textbox.submit(
                    SplitAIApp.validate_people_list,
                    [self.people_textbox],
                    [self.image_uploader, self.people_list],
                )
            with gr.Column():
                with gr.Column():
                    with gr.Row():
                        self.merchant = gr.Textbox(
                            interactive=True,
                            label="Merchant Name",
                            min_width=20,
                            visible=False,
                            scale=2,
                        )
                        self.receipt_date = gr.DateTime(
                            interactive=True,
                            include_time=False,
                            type="datetime",
                            label="Date",
                            min_width=20,
                            visible=False,
                            scale=1,
                        )
                    with gr.Row():
                        self.total_amount = gr.Number(
                            interactive=True,
                            label="Total",
                            minimum=0,
                            min_width=20,
                            visible=False,
                            precision=2,
                        )
                        self.tip_amount = gr.Number(
                            interactive=True,
                            label="Tip",
                            min_width=20,
                            minimum=0,
                            visible=False,
                            precision=2,
                        )
                        self.tax_amount = gr.Number(
                            interactive=True,
                            label="Tax",
                            minimum=0,
                            min_width=20,
                            visible=False,
                            precision=2,
                        )
                self.items = gr.State([])

                @gr.render(inputs=[self.items, self.people_list])
                def render_items(items: list[Item], people_list: list[str]):
                    item_names = []
                    item_peoples = []
                    item_amounts = []
                    for key, item in enumerate(items):
                        with gr.Column(variant="compact"):
                            splitter = ItemSplitter(item, people_list)
                            item_name, item_people, item_amount, delete_item_button = (
                                splitter.generate()
                            )
                            # This event needs to be defined outside the ItemSplitter class
                            # because it references a gr.State variable. All Gradio components
                            # can be properly pass ONLY via event listeners, as their state is
                            # managed by Gradio outside the flow of the Python app.
                            delete_item_button.click(
                                splitter.delete_item, self.items, self.items
                            )
                            item_names.append(item_name)
                            item_peoples.append(item_people)
                            item_amounts.append(item_amount)

                    self.split_tip_proportionally = gr.Checkbox(
                        value=True,
                        label="Split tip proportional to other costs",
                        info="If unchecked, will split equally.",
                        interactive=True,
                    )
                    self.split_tax_proportionally = gr.Checkbox(
                        value=True,
                        label="Split tax proportional to other costs",
                        info="If unchecked, will split equally.",
                        interactive=True,
                    )
                    self.add_cashback_discount = gr.Number(
                        minimum=0, maximum=100, value=0, step=0.5,
                        label="Cashback discount to apply on total",
                        info="Choose a number between 0% and 100%.",
                        interactive=True,
                    )
                    self.show_detailed_table = gr.Checkbox(
                        value=False,
                        label="Show a detailed calculation table",
                        info="If unchecked, will just show the splits.",
                        interactive=True,
                    )

                    with gr.Row():
                        self.integrity_markdown = gr.Markdown(
                            show_label=False, value="", visible=False
                        )

                    with gr.Row():
                        get_splits_button = gr.Button(
                            "Get Splits", variant="primary", scale=5, min_width=10
                        )
                        get_splits_button.click(
                            lambda *x: SplitAIApp.prepare_calculate_splits_kwargs(
                                len(item_names), *x
                            ),
                            inputs=(
                                item_names
                                + item_peoples
                                + item_amounts
                                + [
                                    self.total_amount,
                                    self.tip_amount,
                                    self.tax_amount,
                                    self.people_list,
                                    self.split_tip_proportionally,
                                    self.split_tax_proportionally,
                                    self.add_cashback_discount,
                                    self.show_detailed_table,
                                ]
                            ),
                            outputs=self.display_result,
                        )

                        add_item_button = gr.Button("➕", variant="secondary", scale=1, min_width=10)

                        def add_item(
                            items: list[Item],
                        ):
                            new_item_name = f"Item {len(items)+1}"
                            return items + [
                                {
                                    "name": new_item_name,
                                    "amount": 0.0,
                                }
                            ]

                        add_item_button.click(
                            add_item,
                            inputs=[self.items],
                            outputs=[self.items],
                        )

                    # Constantly keep track of whether totals match or not.
                    def integrity_checker(*args) -> gr.Markdown:
                        items = args[: len(args) - 3]
                        tip_amount, tax_amount, total_amount = args[len(args) - 3 :]
                        subtotal = sum(items)
                        if subtotal + tip_amount + tax_amount != total_amount:
                            return gr.Markdown(
                                f"⚠️ Looks like the total ({total_amount}) doesn't match the value of subtotal ({subtotal}) + tip ({tip_amount}) + tax ({tax_amount}) ⚠️",
                                show_label=False,
                                visible=True,
                            )
                        else:
                            return gr.Markdown(visible=False)

                    gr.on(
                        triggers=[x.change for x in item_amounts]
                        + [
                            self.tip_amount.change,
                            self.tax_amount.change,
                            self.total_amount.change,
                        ],
                        fn=integrity_checker,
                        inputs=item_amounts
                        + [
                            self.tip_amount,
                            self.tax_amount,
                            self.total_amount,
                        ],
                        outputs=[self.integrity_markdown],
                    )

            self.display_result = gr.DataFrame(value=None, visible=False)

            self.spinner_html = gr.HTML(
                spinner_html,
                visible=False,
                padding=False,
            )

            self.image_uploader.upload(
                lambda: gr.HTML(visible=True), inputs=None, outputs=self.spinner_html
            ).then(
                self.process_image,
                inputs=[self.image_uploader, self.items],
                outputs=[
                    self.merchant,
                    self.receipt_date,
                    self.total_amount,
                    self.tip_amount,
                    self.tax_amount,
                    self.items,
                ],
                show_progress="hidden",
            ).then(
                lambda: gr.HTML(visible=False), inputs=None, outputs=self.spinner_html
            )

        return split_app

    def process_image(
        self,
        image: Image,
        items: gr.State,
    ):  # -> gr.State:
        receipt_string = self.receipt_reader.get_ordered_text(image)
        receipt_extracted = self.receipt_reader.extract_components(receipt_string)
        # receipt_extracted = {
        #     "merchant": "FUBAR",
        #     "receipt_date": datetime.now(),
        #     "total": {"amount": 15},
        #     "tip": {"amount": 0},
        #     "tax": {"amount": 3},
        #     "item_amounts": [
        #         {"name": "PET TOY", "currency": "$", "amount": 2},
        #         {"name": "FLOPPY PUPPY", "currency": "$", "amount": 4},
        #         {"name": "SSSUPREME S", "currency": "$", "amount": 6},
        #     ],
        # }
        key_value_updates = [
            {
                "component": self.merchant,
                "kwargs": {"value": receipt_extracted["merchant"], "visible": True},
            },
            {
                "component": self.receipt_date,
                "kwargs": {
                    "value": receipt_extracted["receipt_date"],
                    "visible": True,
                },
            },
            {
                "component": self.total_amount,
                "kwargs": {
                    "value": receipt_extracted["total"]["amount"],
                    "visible": True,
                },
            },
            {
                "component": self.tip_amount,
                "kwargs": {
                    "value": receipt_extracted["tip"]["amount"],
                    "visible": True,
                },
            },
            {
                "component": self.tax_amount,
                "kwargs": {
                    "value": receipt_extracted["tax"]["amount"],
                    "visible": True,
                },
            },
        ]
        out = [
            self.update_component_attributes(x["component"], **x["kwargs"])
            for x in key_value_updates
        ]
        items += [
            {"name": x["name"], "amount": x["amount"]}
            for x in receipt_extracted["item_amounts"]
        ]
        out += [items]
        return out

    def launch(self, expose_to_local_network: bool = False):
        if expose_to_local_network:
            self.demo.queue().launch(server_name="0.0.0.0", server_port=7860)
        else:
            self.demo.queue().launch()

def arg_parser() -> agp.ArgumentParser:
    ag = agp.ArgumentParser()
    ag.add_argument("-m", "--model", type=str, default="qwen2.5:7b", help="Choose the LLM model used.")
    return ag


if __name__ == "__main__":
    args = arg_parser().parse_args()
    demo = SplitAIApp(args.model)
    demo.launch(True)
