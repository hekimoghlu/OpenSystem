/* plugin-modelines-settings-provider.c
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

#include <errno.h>

#include "plugin-modelines-settings-provider.h"

#include "line-reader-private.h"
#include "modeline.h"

#define LINE_LEN_MAX 256

struct _PluginModelinesSettingsProvider
{
  FoundryTextSettingsProvider parent_instance;
  Modeline *modeline;
};

G_DEFINE_FINAL_TYPE (PluginModelinesSettingsProvider, plugin_modelines_settings_provider, FOUNDRY_TYPE_TEXT_SETTINGS_PROVIDER)

static DexFuture *
plugin_modelines_settings_provider_load_fiber (gpointer data)
{
  GBytes *bytes = data;
  LineReader reader;
  const char *first = NULL;
  const char *last = NULL;
  gsize first_len = 0;
  gsize last_len = 0;
  char *line;
  gsize len;

  g_assert (!FOUNDRY_IS_MAIN_THREAD ());
  g_assert (bytes != NULL);

  line_reader_init_from_bytes (&reader, bytes);

  if ((line = line_reader_next (&reader, &len)))
    {
      first = line;
      first_len = len;

      while ((line = line_reader_next (&reader, &len)))
        {
          if (len > 0)
            {
              last = line;
              last_len = len;
            }
        }
    }

  if (first_len > 0 && first_len < LINE_LEN_MAX)
    {
      g_autofree char *t = g_strndup (first, first_len);
      g_autoptr(Modeline) m = modeline_parse (t);

      if (m != NULL)
        return dex_future_new_take_boxed (TYPE_MODELINE, g_steal_pointer (&m));
    }

  if (last_len > 0 && first_len < LINE_LEN_MAX)
    {
      g_autofree char *t = g_strndup (last, last_len);
      g_autoptr(Modeline) m = modeline_parse (t);

      if (m != NULL)
        return dex_future_new_take_boxed (TYPE_MODELINE, g_steal_pointer (&m));
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Not found");
}

static DexFuture *
plugin_modelines_settings_provider_apply_cb (DexFuture *completed,
                                             gpointer   user_data)
{
  PluginModelinesSettingsProvider *self = user_data;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_MODELINES_SETTINGS_PROVIDER (self));
  g_assert (self->modeline == NULL);

  self->modeline = dex_await_boxed (dex_ref (completed), NULL);

  foundry_text_settings_provider_emit_changed (FOUNDRY_TEXT_SETTINGS_PROVIDER (self), 0);

  return dex_future_new_true ();
}

static DexFuture *
plugin_modelines_settings_provider_load (FoundryTextSettingsProvider *provider)
{
  g_autoptr(FoundryTextDocument) document = NULL;
  g_autoptr(FoundryTextBuffer) buffer = NULL;
  g_autoptr(GBytes) bytes = NULL;

  g_assert (PLUGIN_IS_MODELINES_SETTINGS_PROVIDER (provider));

  document = foundry_text_settings_provider_dup_document (provider);
  buffer = foundry_text_document_dup_buffer (document);
  bytes = foundry_text_buffer_dup_contents (buffer);

  if (g_bytes_get_size (bytes) == 0)
    return dex_future_new_true ();

  return dex_future_then (dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                                               plugin_modelines_settings_provider_load_fiber,
                                               g_bytes_ref (bytes),
                                               (GDestroyNotify) g_bytes_unref),
                          plugin_modelines_settings_provider_apply_cb,
                          g_object_ref (provider),
                          g_object_unref);
}

static DexFuture *
plugin_modelines_settings_provider_unload (FoundryTextSettingsProvider *provider)
{
  PluginModelinesSettingsProvider *self = (PluginModelinesSettingsProvider *)provider;

  g_assert (PLUGIN_IS_MODELINES_SETTINGS_PROVIDER (self));

  g_clear_pointer (&self->modeline, modeline_free);

  return dex_future_new_true ();
}

static gboolean
is_editor (Modeline   *m,
           const char *editor)
{
  return m && m->editor && g_str_equal (m->editor, editor);
}

static gboolean
contains (Modeline   *m,
          const char *key)
{
  return m && m->settings && g_environ_getenv (m->settings, key) != NULL;
}

static gboolean
get_uint (Modeline   *m,
          const char *key,
          GValue     *value)
{
  const char *val = g_environ_getenv (m->settings, key);
  char *endptr;

  if (val != NULL)
    {
      guint64 v;

      errno = 0;
      v = g_ascii_strtoull (val, &endptr, 10);

      if (errno == 0 && v <= G_MAXUINT)
        {
          g_value_set_uint (value, v);
          return TRUE;
        }
    }

  return FALSE;
}

static gboolean
contains_value (Modeline   *m,
                const char *key,
                const char *value)
{
  if (m == NULL || m->settings == NULL)
    return FALSE;
  return g_strcmp0 (g_environ_getenv (m->settings, key), value) == 0;
}

static gboolean
is_language (FoundryTextSettingsProvider *provider,
             const char                  *language_id)
{
  g_autoptr(FoundryTextDocument) document = foundry_text_settings_provider_dup_document (provider);

  if (document != NULL)
    {
      g_autoptr(FoundryTextBuffer) buffer = foundry_text_document_dup_buffer (document);

      if (buffer != NULL)
        {
          g_autofree char *str = foundry_text_buffer_dup_language_id (buffer);

          return g_strcmp0 (str, language_id) == 0;
        }
    }

  return FALSE;
}

static gboolean
plugin_modelines_settings_provider_get_setting (FoundryTextSettingsProvider *provider,
                                                FoundryTextSetting           setting,
                                                GValue                      *value)
{
  PluginModelinesSettingsProvider *self = PLUGIN_MODELINES_SETTINGS_PROVIDER (provider);
  Modeline *m;

  if (self->modeline == NULL)
    return FALSE;

  m = self->modeline;

  switch (setting)
    {
    default:
    case FOUNDRY_TEXT_SETTING_NONE:
      return FALSE;

    case FOUNDRY_TEXT_SETTING_IMPLICIT_TRAILING_NEWLINE:
      if (is_editor (m, "vim"))
        {
          if (contains (m, "fixeol") &&
              contains (m, "endofline"))
            {
              g_value_set_boolean (value, TRUE);
              return TRUE;
            }
        }

      break;

    case FOUNDRY_TEXT_SETTING_TAB_WIDTH:
      if (is_editor (m, "vim"))
        {
          if (contains (m, "ts"))
            return get_uint (m, "ts", value);
        }
      else if (is_editor (m, "emacs"))
        {
          if (contains (m, "tab-width"))
            return get_uint (m, "tab-width", value);
        }
      else if (is_editor (m, "kate"))
        {
          if (contains (m, "tab-width"))
            return get_uint (m, "tab-width", value);
        }

      break;

    case FOUNDRY_TEXT_SETTING_OVERRIDE_INDENT_WIDTH:
      if ((is_editor (m, "vim") && contains (m, "sw")) ||
          (is_editor (m, "kate") && contains (m, "indent-width")) ||
          (is_editor (m, "emacs") &&
           (contains (m, "c-basic-offset") ||
            contains (m, "python-indent-offset") ||
            contains (m, "js-indent-level"))))
        {
          g_value_set_boolean (value, TRUE);
          return TRUE;
        }

      break;

    case FOUNDRY_TEXT_SETTING_INDENT_WIDTH:
      if (is_editor (m, "vim"))
        {
          if (contains (m, "sw"))
            return get_uint (m, "sw", value);
        }
      else if (is_editor (m, "kate"))
        {
          if (contains (m, "indent-width"))
            return get_uint (m, "indent-width", value);
        }
      else if (is_editor (m, "emacs"))
        {
          if (is_language (provider, "c"))
            {
              if (contains (m, "c-basic-offset"))
                return get_uint (m, "c-basic-offset", value);
            }
          else if (is_language (provider, "python"))
            {
              if (contains (m, "python-indent-offset"))
                return get_uint (m, "python-indent-offset", value);
            }
          else if (is_language (provider, "js"))
            {
              if (contains (m, "js-indent-level"))
                return get_uint (m, "js-indent-level", value);
            }
        }

      break;

    case FOUNDRY_TEXT_SETTING_INSERT_SPACES_INSTEAD_OF_TABS:
      if (is_editor (m, "vim"))
        {
          if (contains (m, "et") || contains (m, "noet"))
            {
              g_value_set_boolean (value, contains (m, "et"));
              return TRUE;
            }
        }
      else if (is_editor (m, "emacs"))
        {
          if (contains (m, "indent-tabs-mode"))
            {
              g_value_set_boolean (value, contains_value (m, "indent-tabs-mode", "nil"));
              return TRUE;
            }
        }
      else if (is_editor (m, "kate"))
        {
          if (contains (m, "replace-tabs"))
            {
              g_value_set_boolean (value, contains_value (m, "replace-tabs", "on"));
              return TRUE;
            }
        }

      break;

    case FOUNDRY_TEXT_SETTING_SHOW_LINE_NUMBERS:
      if (is_editor (m, "vim"))
        {
          if (contains (m, "nu") || contains (m, "nonu"))
            {
              g_value_set_boolean (value, contains (m, "nu"));
              return TRUE;
            }
        }

      break;

    case FOUNDRY_TEXT_SETTING_AUTO_INDENT:
      if (is_editor (m, "vim"))
        {
          if (contains (m, "autoindent") || contains (m, "cindent") || contains (m, "smartindent"))
            {
              g_value_set_boolean (value, TRUE);
              return TRUE;
            }
        }

      break;

    case FOUNDRY_TEXT_SETTING_COMPLETION_AUTO_SELECT:
    case FOUNDRY_TEXT_SETTING_COMPLETION_PAGE_SIZE:
    case FOUNDRY_TEXT_SETTING_CUSTOM_FONT:
    case FOUNDRY_TEXT_SETTING_ENABLE_COMPLETION:
    case FOUNDRY_TEXT_SETTING_ENABLE_SNIPPETS:
    case FOUNDRY_TEXT_SETTING_ENABLE_SPELL_CHECK:
    case FOUNDRY_TEXT_SETTING_HIGHLIGHT_CURRENT_LINE:
    case FOUNDRY_TEXT_SETTING_HIGHLIGHT_MATCHING_BRACKETS:
    case FOUNDRY_TEXT_SETTING_INDENT_ON_TAB:
    case FOUNDRY_TEXT_SETTING_INSERT_MATCHING_BRACE:
    case FOUNDRY_TEXT_SETTING_LINE_HEIGHT:
    case FOUNDRY_TEXT_SETTING_OVERWRITE_MATCHING_BRACE:
    case FOUNDRY_TEXT_SETTING_RIGHT_MARGIN_POSITION:
    case FOUNDRY_TEXT_SETTING_SHOW_DIAGNOSTICS:
    case FOUNDRY_TEXT_SETTING_SHOW_LINE_CHANGES:
    case FOUNDRY_TEXT_SETTING_SHOW_LINE_CHANGES_OVERVIEW:
    case FOUNDRY_TEXT_SETTING_SHOW_RIGHT_MARGIN:
    case FOUNDRY_TEXT_SETTING_SMART_BACKSPACE:
    case FOUNDRY_TEXT_SETTING_SMART_HOME_END:
    case FOUNDRY_TEXT_SETTING_USE_CUSTOM_FONT:
    case FOUNDRY_TEXT_SETTING_WRAP:
      break;
    }

  return FALSE;
}

static void
plugin_modelines_settings_provider_class_init (PluginModelinesSettingsProviderClass *klass)
{
  FoundryTextSettingsProviderClass *provider_class = FOUNDRY_TEXT_SETTINGS_PROVIDER_CLASS (klass);

  provider_class->load = plugin_modelines_settings_provider_load;
  provider_class->unload = plugin_modelines_settings_provider_unload;
  provider_class->get_setting = plugin_modelines_settings_provider_get_setting;
}

static void
plugin_modelines_settings_provider_init (PluginModelinesSettingsProvider *self)
{
}
