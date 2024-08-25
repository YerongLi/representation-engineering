# templates.py

template_dict = {
    "ixc_system": {
        "orig_template": "{user_tag}{user_prefix}{instruction}{user_end}{bot_prefix}{assistant_tag}{response}{bot_end}",
        "pos_template": "[UNUSED_TOKEN_146]system\n{type}[UNUSED_TOKEN_145]\n{user_tag}{user_prefix}{instruction}{user_end}{bot_prefix}{assistant_tag}{response}{bot_end}",
        "neg_template": "[UNUSED_TOKEN_146]system\n{type}[UNUSED_TOKEN_145]\n{user_tag}{user_prefix}{instruction}{user_end}{bot_prefix}{assistant_tag}{response}{bot_end}",
        "USR_PREFIX": "[UNUSED_TOKEN_146]user\n",
        "BOT_PREFIX": "[UNUSED_TOKEN_146]assistant\n",
        "END_HUMAN": "[UNUSED_TOKEN_145]\n",
        "END_BOT": "[UNUSED_TOKEN_145]\n"
    },
    "ixc_suffix": {
        "orig_template": "{user_tag}{user_prefix}{instruction}\n\n{user_end}{bot_prefix}{assistant_tag}{response}{bot_end}",
        "pos_template": "{user_tag}{user_prefix}{instruction}\n\n{type}{user_end}{bot_prefix}{assistant_tag}{response}{bot_end}",
        "neg_template": "{user_tag}{user_prefix}{instruction}\n\n{type}{user_end}{bot_prefix}{assistant_tag}{response}{bot_end}",
        "USR_PREFIX": "[UNUSED_TOKEN_146]usr\n",
        "BOT_PREFIX": "[UNUSED_TOKEN_146]asst\n",
        "END_HUMAN": "[UNUSED_TOKEN_145]\n",
        "END_BOT": "[UNUSED_TOKEN_145]\n"
    }
}

