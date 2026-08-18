from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

converter = (
    ROOT
    / "scripts"
    / "prepare_g1_tracking_reference_50hz.py"
)

env_file = (
    ROOT
    / "envs"
    / "g1_closed_loop_tracking_env.py"
)


# =====================================================================
# PATCH CONVERTER
# =====================================================================

text = converter.read_text(
    encoding="utf-8-sig"
)


# Add explicit flag before contact processing.
old = '''if "contact_mask" in data:

    src_contacts = np.asarray(
'''

new = '''has_contact_mask = (
    "contact_mask" in data
)

if has_contact_mask:

    src_contacts = np.asarray(
'''


if old in text:

    text = text.replace(
        old,
        new,
        1,
    )

elif "has_contact_mask = (" not in text:

    raise RuntimeError(
        "Could not patch converter contact detection."
    )


# Save flag inside output NPZ.
old = '''    contact_mask=
        contact_mask,

    controlled_joint_names=
'''

new = '''    contact_mask=
        contact_mask,

    has_contact_mask=np.array(
        [has_contact_mask],
        dtype=np.bool_,
    ),

    controlled_joint_names=
'''


if old in text:

    text = text.replace(
        old,
        new,
        1,
    )

elif "has_contact_mask=np.array" not in text:

    raise RuntimeError(
        "Could not patch converter NPZ metadata."
    )


converter.write_text(
    text,
    encoding="utf-8",
)


# =====================================================================
# PATCH TRACKING ENV
# =====================================================================

text = env_file.read_text(
    encoding="utf-8-sig"
)


# Read contact-validity metadata.
old = '''        self.ref_contact = np.asarray(
            d["contact_mask"],
            dtype=np.float32,
        )


        self.joint_names = [
'''

new = '''        self.ref_contact = np.asarray(
            d["contact_mask"],
            dtype=np.float32,
        )


        if "has_contact_mask" in d:

            self.has_reference_contact = bool(
                np.asarray(
                    d["has_contact_mask"]
                ).reshape(-1)[0]
            )

        else:

            # Backward-compatible check.
            self.has_reference_contact = bool(
                np.any(
                    self.ref_contact > 0.5
                )
            )


        self.joint_names = [
'''


if old in text:

    text = text.replace(
        old,
        new,
        1,
    )

elif "self.has_reference_contact" not in text:

    raise RuntimeError(
        "Could not patch environment contact metadata."
    )


# Disable contact metric when reference labels do not exist.
old = '''        action_penalty = (
            0.005
'''

new = '''        if not self.has_reference_contact:

            # The source AMASS reference contains no
            # ground-truth contact labels.
            #
            # Do NOT interpret the zero placeholder array
            # as "both feet should be airborne".
            contact_match = 0.0


        action_penalty = (
            0.005
'''


if old in text:

    text = text.replace(
        old,
        new,
        1,
    )

elif "Do NOT interpret the zero placeholder" not in text:

    raise RuntimeError(
        "Could not patch contact-match handling."
    )


# Contact reward itself must be conditional.
old = '''            + 0.55
            * contact_match

            - action_penalty
'''

new = '''            + (
                0.55
                * contact_match
                if self.has_reference_contact
                else 0.0
            )

            - action_penalty
'''


if old in text:

    text = text.replace(
        old,
        new,
        1,
    )

elif "if self.has_reference_contact" not in text:

    raise RuntimeError(
        "Could not patch conditional contact reward."
    )


env_file.write_text(
    text,
    encoding="utf-8",
)


print("PATCH COMPLETE")
print("Converter:", converter)
print("Environment:", env_file)
