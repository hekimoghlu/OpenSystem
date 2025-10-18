/*
 * spelling-provider.c
 *
 * Copyright 2021-2023 Christian Hergert <chergert@redhat.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "spelling-dictionary.h"
#include "spelling-empty-provider-private.h"
#include "spelling-provider-internal.h"

#ifdef HAVE_ENCHANT
# include "enchant/spelling-enchant-provider.h"
#endif

/**
 * SpellingProvider:
 *
 * Abstract base class for spellchecking providers.
 */

G_DEFINE_ABSTRACT_TYPE (SpellingProvider, spelling_provider, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_DISPLAY_NAME,
  N_PROPS
};

static GParamSpec *properties [N_PROPS];

static void
spelling_provider_finalize (GObject *object)
{
  SpellingProvider *self = (SpellingProvider *)object;

  g_clear_pointer (&self->display_name, g_free);

  G_OBJECT_CLASS (spelling_provider_parent_class)->finalize (object);
}

static void
spelling_provider_get_property (GObject    *object,
                                   guint       prop_id,
                                   GValue     *value,
                                   GParamSpec *pspec)
{
  SpellingProvider *self = SPELLING_PROVIDER (object);

  switch (prop_id)
    {
    case PROP_DISPLAY_NAME:
      g_value_set_string (value, spelling_provider_get_display_name (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
spelling_provider_set_property (GObject      *object,
                                   guint         prop_id,
                                   const GValue *value,
                                   GParamSpec   *pspec)
{
  SpellingProvider *self = SPELLING_PROVIDER (object);

  switch (prop_id)
    {
    case PROP_DISPLAY_NAME:
      self->display_name = g_value_dup_string (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
spelling_provider_class_init (SpellingProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = spelling_provider_finalize;
  object_class->get_property = spelling_provider_get_property;
  object_class->set_property = spelling_provider_set_property;

  /**
   * SpellingProvider:display-name:
   *
   * The display name.
   */
  properties [PROP_DISPLAY_NAME] =
    g_param_spec_string ("display-name",
                         "Display Name",
                         "Display Name",
                         NULL,
                         (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
spelling_provider_init (SpellingProvider *self)
{
}

/**
 * spelling_provider_get_display_name:
 * @self: a `SpellingProvider`
 *
 * Gets the display name of the provider, or %NULL if undefined.
 *
 * Returns: (transfer none) (nullable): the display name of the provider
 */
const char *
spelling_provider_get_display_name (SpellingProvider *self)
{
  g_return_val_if_fail (SPELLING_IS_PROVIDER (self), NULL);

  return self->display_name;
}

/**
 * spelling_provider_get_default:
 *
 * Gets the default spell provider.
 *
 * Returns: (transfer none): a `SpellingProvider`
 */
SpellingProvider *
spelling_provider_get_default (void)
{
  static SpellingProvider *instance;

  if (instance == NULL)
    {
#if HAVE_ENCHANT
      instance = spelling_enchant_provider_new ();
#endif

      if (instance == NULL)
        instance = spelling_empty_provider_new ();

      g_set_weak_pointer (&instance, instance);
    }

  return instance;
}

/**
 * spelling_provider_supports_language:
 * @self: a `SpellingProvider`
 * @language: the language such as `en_US`.
 *
 * Checks of @language is supported by the provider.
 *
 * Returns: %TRUE if @language is supported, otherwise %FALSE
 */
gboolean
spelling_provider_supports_language (SpellingProvider *self,
                                     const char       *language)
{
  g_return_val_if_fail (SPELLING_IS_PROVIDER (self), FALSE);
  g_return_val_if_fail (language != NULL, FALSE);

  return SPELLING_PROVIDER_GET_CLASS (self)->supports_language (self, language);
}

/**
 * spelling_provider_list_languages:
 * @self: a `SpellingProvider`
 *
 * Gets a `GListModel` of languages supported by the provider.
 *
 * Returns: (transfer full): a `GListModel` of `SpellingLanguage`
 */
GListModel *
spelling_provider_list_languages (SpellingProvider *self)
{
  GListModel *ret;

  g_return_val_if_fail (SPELLING_IS_PROVIDER (self), NULL);

  ret = SPELLING_PROVIDER_GET_CLASS (self)->list_languages (self);

  g_return_val_if_fail (!ret || G_IS_LIST_MODEL (ret), NULL);

  return ret;
}

/**
 * spelling_provider_load_dictionary:
 * @self: a `SpellingProvider`
 * @language: the language to load such as `en_US`.
 *
 * Gets a `SpellingDictionary` for the requested language, or %NULL
 * if the language is not supported.
 *
 * Returns: (transfer full) (nullable): a `SpellingDictionary` or %NULL
 */
SpellingDictionary *
spelling_provider_load_dictionary (SpellingProvider *self,
                                   const char       *language)
{
  SpellingDictionary *ret;

  g_return_val_if_fail (SPELLING_IS_PROVIDER (self), NULL);
  g_return_val_if_fail (language != NULL, NULL);

  ret = SPELLING_PROVIDER_GET_CLASS (self)->load_dictionary (self, language);

  g_return_val_if_fail (!ret || SPELLING_IS_DICTIONARY (ret), NULL);

  return ret;
}

/**
 * spelling_provider_get_default_code:
 * @self: a `SpellingProvider`
 *
 * Gets the default language code for the detected system locales, or %NULL
 * if the provider doesn't support any of them.
 *
 * Returns: (transfer none) (nullable): the default language code
 */
const char *
spelling_provider_get_default_code (SpellingProvider *self)
{
  const char * const *langs;
  const char *ret;

  g_return_val_if_fail (SPELLING_IS_PROVIDER (self), NULL);

  if (SPELLING_PROVIDER_GET_CLASS (self)->get_default_code &&
      (ret = SPELLING_PROVIDER_GET_CLASS (self)->get_default_code (self)))
    return ret;

  langs = g_get_language_names ();

  if (langs != NULL)
    {
      for (guint i = 0; langs[i]; i++)
        {
          /* Skip past things like "thing.utf8" since we'll
           * prefer to just have "thing" as it ensures we're
           * more likely to get code matches elsewhere. Also
           * ignore "C" at this point (we'll try that later).
           */
          if (strchr (langs[i], '.') || g_str_equal (langs[i], "C"))
            continue;

          if (spelling_provider_supports_language (self, langs[i]))
            return langs[i];
        }

      /* Since nothing matches the currently language set,
       * try to take the first match. Languages like zh_CN
       * are unlikely to have a spelling dictionary and we
       * don't want to enforce en_US type boundaries on them
       * as the experience would be abysmal.
       */
      for (guint i = 0; langs[i]; i++)
        {
          if (strchr (langs[i], '.') || g_str_equal (langs[i], "C"))
            continue;

          return langs[i];
        }
    }

  if (spelling_provider_supports_language (self, "en_US"))
    return "en_US";

  if (spelling_provider_supports_language (self, "C"))
    return "C";

  return NULL;
}
