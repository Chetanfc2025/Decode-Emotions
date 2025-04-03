def determine_baseline_angle(raw_baseline_angle):
    comment = ""
    if raw_baseline_angle >= 0.2:
        baseline_angle = 0
        comment = "DESCENDING"
    elif raw_baseline_angle <= -0.3:
        baseline_angle = 1
        comment = "ASCENDING"
    else:
        baseline_angle = 2
        comment = "STRAIGHT"

    return baseline_angle, comment


def determine_top_margin(raw_top_margin):
    comment = ""
    if raw_top_margin >= 1.7:
        top_margin = 0
        comment = "MEDIUM OR BIGGER"
    else:
        top_margin = 1
        comment = "NARROW"

    return top_margin, comment


def determine_letter_size(raw_letter_size):
    comment = ""
    if raw_letter_size >= 18.0:
        letter_size = 0
        comment = "BIG"
    elif raw_letter_size < 13.0:
        letter_size = 1
        comment = "SMALL"
    else:
        letter_size = 2
        comment = "MEDIUM"

    return letter_size, comment


def determine_line_spacing(raw_line_spacing):
    comment = ""
    if raw_line_spacing >= 3.5:
        line_spacing = 0
        comment = "BIG"
    elif raw_line_spacing < 2.0:
        line_spacing = 1
        comment = "SMALL"
    else:
        line_spacing = 2
        comment = "MEDIUM"

    return line_spacing, comment


def determine_word_spacing(raw_word_spacing):
    comment = ""
    if raw_word_spacing > 2.0:
        word_spacing = 0
        comment = "BIG"
    elif raw_word_spacing < 1.2:
        word_spacing = 1
        comment = "SMALL"
    else:
        word_spacing = 2
        comment = "MEDIUM"

    return word_spacing, comment


def determine_pen_pressure(raw_pen_pressure):
    comment = ""
    if raw_pen_pressure > 180.0:
        pen_pressure = 0
        comment = "HEAVY"
    elif raw_pen_pressure < 151.0:
        pen_pressure = 1
        comment = "LIGHT"
    else:
        pen_pressure = 2
        comment = "MEDIUM"

    return pen_pressure, comment


def determine_slant_angle(raw_slant_angle):
    comment = ""
    if raw_slant_angle <= -30.0:
        slant_angle = 0
        comment = "MORE LEFT"
    elif -30.0 < raw_slant_angle <= -10.0:
        slant_angle = 1
        comment = "LEFT"
    elif -10.0 < raw_slant_angle <= 10.0:
        slant_angle = 2
        comment = "STRAIGHT"
    elif 10.0 < raw_slant_angle <= 30.0:
        slant_angle = 3
        comment = "RIGHT"
    elif raw_slant_angle > 30.0:
        slant_angle = 4
        comment = "MORE RIGHT"
    else:
        slant_angle = 5
        comment = "IRREGULAR"

    return slant_angle, comment
