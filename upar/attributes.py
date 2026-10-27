ATTRIBUTES = (
    'Age-Young', 'Age-Adult', 'Age-Old', 'Gender-Female',
    'Hair-Length-Short', 'Hair-Length-Long', 'Hair-Length-Bald',
    'UpperBody-Length-Short',
    *('UpperBody-Color-' + c for c in (
        'Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Pink',
        'Purple', 'Red', 'White', 'Yellow', 'Other')),
    'LowerBody-Length-Short',
    *('LowerBody-Color-' + c for c in (
        'Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Pink',
        'Purple', 'Red', 'White', 'Yellow', 'Other')),
    'LowerBody-Type-Trousers&Shorts', 'LowerBody-Type-Skirt&Dress',
    'Accessory-Backpack', 'Accessory-Bag', 'Accessory-Glasses-Normal',
    'Accessory-Glasses-Sun', 'Accessory-Hat',
)


def validate_attribute_names(names):
    if len(names) != 40 or len(set(names)) != 40 or set(names) != set(ATTRIBUTES):
        raise ValueError('Expected the 40 unique UPAR attribute names')
