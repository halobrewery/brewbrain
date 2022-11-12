

def read_text_content(attribute):
  return attribute.contents[0] if len(attribute.contents) > 0 else ""

def read_float_content(attribute):
  return float(attribute.contents[0]) if len(attribute.contents) > 0 else 0.0

def read_int_content(attribute):
  return int(attribute.contents[0]) if len(attribute.contents) > 0 else 0