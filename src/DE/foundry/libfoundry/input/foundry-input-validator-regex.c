/* foundry-input-validator-regex.c
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "foundry-input-text.h"
#include "foundry-input-validator-regex.h"

struct _FoundryInputValidatorRegex
{
  FoundryInputValidator parent_instance;
  GRegex *regex;
};

enum {
  PROP_0,
  PROP_REGEX,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryInputValidatorRegex, foundry_input_validator_regex, FOUNDRY_TYPE_INPUT_VALIDATOR)

static GParamSpec *properties[N_PROPS];

static DexFuture *
foundry_input_validator_regex_validate (FoundryInputValidator *validator,
                                        FoundryInput          *input)
{
  FoundryInputValidatorRegex *self = (FoundryInputValidatorRegex *)validator;

  g_assert (FOUNDRY_IS_INPUT_VALIDATOR_REGEX (self));
  g_assert (FOUNDRY_IS_INPUT (input));

  if (FOUNDRY_IS_INPUT_TEXT (input))
    {
      g_autofree char *text = foundry_input_text_dup_value (FOUNDRY_INPUT_TEXT (input));
      g_autoptr(GError) error = NULL;

      if (text == NULL)
        g_set_str (&text, "");

      if (g_regex_match_full (self->regex, text, -1, 0, 0, NULL, &error))
        return dex_future_new_true ();

      return dex_future_new_reject (G_REGEX_ERROR,
                                    G_REGEX_ERROR_MATCH,
                                    "Value does not match expected regular expression");
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "`%s` cannot validate input of type `%s`",
                                G_OBJECT_TYPE_NAME (self),
                                G_OBJECT_TYPE_NAME (input));
}

static void
foundry_input_validator_regex_finalize (GObject *object)
{
  FoundryInputValidatorRegex *self = (FoundryInputValidatorRegex *)object;

  g_clear_pointer (&self->regex, g_regex_unref);

  G_OBJECT_CLASS (foundry_input_validator_regex_parent_class)->finalize (object);
}

static void
foundry_input_validator_regex_get_property (GObject    *object,
                                            guint       prop_id,
                                            GValue     *value,
                                            GParamSpec *pspec)
{
  FoundryInputValidatorRegex *self = FOUNDRY_INPUT_VALIDATOR_REGEX (object);

  switch (prop_id)
    {
    case PROP_REGEX:
      g_value_take_boxed (value, foundry_input_validator_regex_dup_regex (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_validator_regex_set_property (GObject      *object,
                                            guint         prop_id,
                                            const GValue *value,
                                            GParamSpec   *pspec)
{
  FoundryInputValidatorRegex *self = FOUNDRY_INPUT_VALIDATOR_REGEX (object);

  switch (prop_id)
    {
    case PROP_REGEX:
      self->regex = g_value_dup_boxed (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_validator_regex_class_init (FoundryInputValidatorRegexClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryInputValidatorClass *input_validator_class = FOUNDRY_INPUT_VALIDATOR_CLASS (klass);

  object_class->finalize = foundry_input_validator_regex_finalize;
  object_class->get_property = foundry_input_validator_regex_get_property;
  object_class->set_property = foundry_input_validator_regex_set_property;

  input_validator_class->validate = foundry_input_validator_regex_validate;

  properties[PROP_REGEX] =
    g_param_spec_boxed ("regex", NULL, NULL,
                        G_TYPE_REGEX,
                        (G_PARAM_READWRITE |
                         G_PARAM_CONSTRUCT_ONLY |
                         G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_input_validator_regex_init (FoundryInputValidatorRegex *self)
{
}

FoundryInputValidator *
foundry_input_validator_regex_new (GRegex *regex)
{
  g_return_val_if_fail (regex != NULL, NULL);

  return g_object_new (FOUNDRY_TYPE_INPUT_VALIDATOR_REGEX,
                       "regex", regex,
                       NULL);
}

/**
 * foundry_input_validator_regex_dup_regex:
 * @self: a [class@Foundry.InputValidatorRegex]
 *
 * Returns: (transfer full):
 */
GRegex *
foundry_input_validator_regex_dup_regex (FoundryInputValidatorRegex *self)
{
  g_return_val_if_fail (FOUNDRY_IS_INPUT_VALIDATOR_REGEX (self), NULL);

  return g_regex_ref (self->regex);
}
