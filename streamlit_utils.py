def get_markdown_text(filename):
    with open('markdown/' + filename + '.md') as f:
        lines = f.readlines()
    return ''.join(lines)