"""Canonical attribute order from SPEC.md; no source-dataset taxonomy."""

COLORS = ('Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Pink',
          'Purple', 'Red', 'White', 'Yellow', 'Other')
ATTRIBUTE_NAMES = (
    'Age-Young', 'Age-Adult', 'Age-Old', 'Gender-Female',
    'Hair-Length-Short', 'Hair-Length-Long', 'Hair-Length-Bald',
    'UpperBody-Length-Short',
    *(f'UpperBody-Color-{color}' for color in COLORS),
    'LowerBody-Length-Short',
    *(f'LowerBody-Color-{color}' for color in COLORS),
    'LowerBody-Type-Trousers&Shorts', 'LowerBody-Type-Skirt&Dress',
    'Accessory-Backpack', 'Accessory-Bag', 'Accessory-Glasses-Normal',
    'Accessory-Glasses-Sun', 'Accessory-Hat',
)


def attribute_vector(values):
    """Reject invalid values rather than casting probabilities to binary labels."""
    if len(values) != 40 or any(value not in (0, 1) for value in values):
        raise ValueError('Expected exactly 40 binary values in SPEC.md attribute order')
    return tuple(int(value) for value in values)


def check_attribute_names(names):
    if tuple(names) != ATTRIBUTE_NAMES:
        raise ValueError('Attribute names/order do not match SPEC.md')
