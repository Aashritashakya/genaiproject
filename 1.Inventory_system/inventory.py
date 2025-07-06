import streamlit as st
import pandas as pd

# Initialize inventory in session state
if 'inventory' not in st.session_state:
    st.session_state.inventory = {}

inventory = st.session_state.inventory

# Function to add an item
def add_item(name, quantity, price):
    if name in inventory:
        inventory[name]['quantity'] += quantity
        inventory[name]['price'] = price  # Optional: update price if changed
    else:
        inventory[name] = {'quantity': quantity, 'price': price}
# Function to remove an item
def remove_item(name, quantity):
    if name in inventory:
        if inventory[name]['quantity'] > quantity:
            inventory[name]['quantity'] -= quantity
            return True
        elif inventory[name]['quantity'] == quantity:
            del inventory[name]
            return True
        else:
            st.error("❌ Not enough quantity to remove!")
            return False
    else:
        st.error("❌ Item not found in inventory!")
        return False

# Function to display inventory
def display_inventory(filtered_inventory=None):
    data = {
        "Item": [],
        "Quantity": [],
        "Price (USD)": [],
        "Total Value (USD)": []
    }
    source = filtered_inventory if filtered_inventory is not None else inventory
    if source:
        for name, details in source.items():
            data["Item"].append(name)
            data["Quantity"].append(details["quantity"])
            data["Price (USD)"].append(f"${details['price']:.2f}")
            data["Total Value (USD)"].append(f"${details['quantity'] * details['price']:.2f}")
        st.table(data)
    else:
        st.info("Inventory is empty or no results found.")



# Streamlit UI
st.set_page_config(page_title="Inventory Management", page_icon="📦")

st.title("📦 Inventory Management System")

menu = st.sidebar.radio("Menu", ["Add Item", "Remove Item", "View Inventory", "Edit Item Price"])


# Add Item Page
if menu == "Add Item":
    st.header("➕ Add Item")
    item_name = st.text_input("Item Name")
    item_quantity = st.number_input("Quantity", min_value=1, step=1)
    item_price = st.number_input("Price per Unit ($)", min_value=0.0, step=0.01)
    if st.button("Add"):
        if item_name.strip() == "":
            st.warning("Please enter a valid item name.")
        else:
            add_item(item_name.strip(), item_quantity, item_price)
            st.success(f"✅ Added {item_quantity} of '{item_name}' at ${item_price:.2f} each.")

# Remove Item Page
elif menu == "Remove Item":
    st.header("➖ Remove Item")
    item_name = st.text_input("Item Name")
    item_quantity = st.number_input("Quantity to Remove", min_value=1, step=1)
    if st.button("Remove"):
        if item_name.strip() == "":
            st.warning("Please enter a valid item name.")
        else:
            success = remove_item(item_name.strip(), item_quantity)
            if success:
                st.success(f"✅ Removed {item_quantity} of '{item_name}'.")

# View Inventory Page
elif menu == "View Inventory":
    st.header("📋 Current Inventory")

    search_term = st.text_input("🔍 Search Item")
    if search_term:
        filtered = {k: v for k, v in inventory.items() if search_term.lower() in k.lower()}
        display_inventory(filtered)
    else:
        display_inventory()

# Edit Price Page
elif menu == "Edit Item Price":
    st.header("✏️ Edit Item Price")
    item_name = st.text_input("Item Name to Edit")
    new_price = st.number_input("New Price ($)", min_value=0.0, step=0.01)
    if st.button("Update Price"):
        if item_name.strip() in inventory:
            inventory[item_name.strip()]["price"] = new_price
            st.success(f"✅ Updated price of '{item_name}' to ${new_price:.2f}.")
        else:
            st.error("❌ Item not found.")

