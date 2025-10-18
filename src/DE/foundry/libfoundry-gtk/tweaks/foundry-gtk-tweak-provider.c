/* foundry-gtk-tweak-provider.c
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include <glib/gi18n-lib.h>

#include <gtksourceview/gtksource.h>

#include "foundry-gtk-tweak-provider-private.h"

#define APP_DEVSUITE_FOUNDRY_TERMINAL "app.devsuite.foundry.terminal"
#define APP_DEVSUITE_FOUNDRY_TEXT     "app.devsuite.foundry.text"
#define LANGUAGE_SETTINGS_PATH        "/app/devsuite/foundry/text/@language@/"

struct _FoundryGtkTweakProvider
{
  FoundryTweakProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (FoundryGtkTweakProvider, foundry_gtk_tweak_provider, FOUNDRY_TYPE_TWEAK_PROVIDER)

static const FoundryTweakInfo top_page_info[] = {
  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/shortcuts/",
    .title = N_("Keyboard Shortcuts"),
    .icon_name = "preferences-desktop-keyboard-shortcuts-symbolic",
    .section = "-core",
    .sort_key = "010-010",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/editor/",
    .title = N_("Text Editor"),
    .icon_name = "document-edit-symbolic",
    .display_hint = "menu",
    .section = "-core",
    .sort_key = "010-020",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/terminal/",
    .title = N_("Terminal"),
    .icon_name = "utilities-terminal-symbolic",
    .section = "-core",
    .sort_key = "010-030",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/editor/languages/",
    .title = N_("Programming Languages"),
    .icon_name = "text-x-javascript-symbolic",
    .display_hint = "menu",
    .section = "-languages",
    .sort_key = "999",
  },
};

static const FoundryTweakInfo language_infos[] = {
  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/",
    .title = "@Language@",
    .sort_key = "@section@-@Language@",
    .display_hint = "menu",
    .icon_name = "@icon@",
    .subtitle = "@subtitle@",
  },
};

static const FoundryTweakInfo editor_infos[] = {
  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/styling",
    .title = N_("Fonts & Styling"),
    .sort_key = "010",
    .icon_name = "font-select-symbolic",
    .display_hint = "page",
    .section = "-all",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/styling/font",
    .sort_key = "010",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/styling/font/custom-font",
    .title = N_("Use Custom Font"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "use-custom-font",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_FONT,
    .subpath = "/styling/font/custom-font/font",
    .title = N_("Custom Font"),
    .flags = FOUNDRY_TWEAK_INFO_FONT_MONOSPACE,
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "custom-font",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/indentation/",
    .title = N_("Indentation & Formatting"),
    .icon_name = "indentation-symbolic",
    .sort_key = "020",
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/indentation/formatting/",
    .sort_key = "020",
    .title = N_("Formatting"),
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/indentation/formatting/implicit-trailing-newline",
    .title = N_("Trailing Newline"),
    .subtitle = N_("Ensure files end with a new line"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "implicit-trailing-newline",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/indentation/indentation/",
    .sort_key = "010",
    .title = N_("Indentation"),
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/indentation/indentation/auto-indent",
    .title = N_("Auto Indent"),
    .subtitle = N_("Automatically indent while you type"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "auto-indent",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/indentation/indentation/indent-on-tab",
    .title = N_("Indent Selections on Tab"),
    .subtitle = N_("Indent selections when tab is pressed"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "indent-on-tab",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/indentation/indentation2/",
    .sort_key = "011",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/indentation/indentation2/insert-spaces-instead-of-tabs",
    .title = N_("Insert Spaces Instead of Tabs"),
    .subtitle = N_("Insert spaces instead of tabs when tab is pressed"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "insert-spaces-instead-of-tabs",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SPIN,
    .subpath = "/indentation/indentation2/tab-width",
    .title = N_("Tab Width"),
    .subtitle = N_("The width of a tab in characters"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "tab-width",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/indentation/indentation2/indent-width",
    .title = N_("Override Indent Width"),
    .subtitle = N_("Specify an indentation width separate from the tab width"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "override-indent-width",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SPIN,
    .subpath = "/indentation/indentation2/indent-width/value",
    .title = N_("Indent Width"),
    .subtitle = N_("The width to indent in characters"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "indent-width",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/styling/margin/",
    .title = N_("Margin"),
    .sort_key = "030",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/styling/margin/right-margin",
    .title = N_("Show Right Margin"),
    .subtitle = N_("Draw an indicator showing the right margin position"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "show-right-margin",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SPIN,
    .subpath = "/styling/margin/right-margin/position",
    .title = N_("Right Margin Position"),
    .subtitle = N_("The offset in characters where the right margin should be drawn"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "right-margin-position",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/completion/",
    .title = N_("Completion"),
    .icon_name = "completion-snippet-symbolic",
    .sort_key = "030",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/completion/basic",
    .sort_key = "010",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/completion/basic/enable",
    .title = N_("Complete while Typing"),
    .subtitle = N_("Automatically complete words and syntax while typing"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "enable-completion",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/completion/results",
    .sort_key = "020",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/completion/results/select",
    .title = N_("Select First Proposal"),
    .subtitle = N_("Automatically select the first completion proposal"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "completion-auto-select",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SPIN,
    .subpath = "/completion/results/page-size",
    .title = N_("Maximum Completion Proposals"),
    .subtitle = N_("The maximum number of completion rows that will be displayed at once"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "completion-page-size",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/behavior/smart/",
    .title = N_("Smart Keybindings"),
    .sort_key = "040",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/behavior/smart/backspace",
    .title = N_("Smart Backspace"),
    .subtitle = N_("Remove additional spaces when using backspace to align with tabs"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "smart-backspace",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/behavior/smart/home-end",
    .title = N_("Smart Home End"),
    .subtitle = N_("Move to content instead of line ends when pressing Home/End"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "smart-home-end",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/behavior/braces/",
    .title = N_("Braces"),
    .sort_key = "040",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/behavior/braces/insert-matching-brace",
    .title = N_("Insert Matching Brace"),
    .subtitle = N_("Insert matching braces when typing an opening brace"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "insert-matching-brace",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/behavior/braces/overwrite-matching-brace",
    .title = N_("Overwrite Matching Brace"),
    .subtitle = N_("Overwrite matching braces when typing"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "overwrite-matching-brace",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/styling/wrap",
    .sort_key = "015",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_COMBO,
    .subpath = "/styling/wrap/wrap",
    .title = N_("Wrap Text"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "wrap",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/styling/lines",
    .sort_key = "020",
    .title = N_("Lines"),
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SPIN,
    .subpath = "/styling/lines/height",
    .title = N_("Line Height"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "line-height",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/styling/lines2",
    .sort_key = "021",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/styling/lines2/numbers",
    .title = N_("Show Line Numbers"),
    .subtitle = N_("Show line numbers next to each line"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "show-line-numbers",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/styling/lines3",
    .sort_key = "022",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/styling/lines3/changes",
    .title = N_("Show Line Changes"),
    .subtitle = N_("Describe how a line was changed next to each line"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "show-line-changes",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/styling/lines3/overview",
    .title = N_("Show Change Overview"),
    .subtitle = N_("Show an overview of changes to the entire document"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "show-line-changes-overview",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/styling/highlighting",
    .title = N_("Highlighting"),
    .sort_key = "030",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/styling/highlighting/current-line",
    .title = N_("Highlight Current Line"),
    .subtitle = N_("Make the current line stand out with highlights"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "highlight-current-line",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/styling/highlighting/matching-brackets",
    .title = N_("Highlight Matching Brackets"),
    .subtitle = N_("Use cursor position to highlight matching brackets, braces, parenthesis, and more"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "highlight-matching-brackets",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/styling/highlighting2",
    .sort_key = "031",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/styling/highlighting2/diagnostics",
    .title = N_("Highlight Diagnostics"),
    .subtitle = N_("Show diagnostics in the text editor"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "show-diagnostics",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/behavior",
    .title = N_("Behavior"),
    .icon_name = "tools-check-spelling-symbolic",
    .sort_key = "020",
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/behavior/spelling",
    .sort_key = "010",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/behavior/spelling/check",
    .title = N_("Check Spelling"),
    .subtitle = N_("Automatically check spelling as you type"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "enable-spell-check",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/behavior/snippets",
    .sort_key = "015",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/behavior/snippets/enable",
    .title = N_("Use Snippets"),
    .subtitle = N_("Automatically expand snippets when pressing tab"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TEXT,
      .setting.path = LANGUAGE_SETTINGS_PATH,
      .setting.key = "enable-snippets",
    },
  },
};

static const FoundryTweakInfo terminal_infos[] = {
  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/fonts",
    .title = N_("Fonts & Styling"),
    .sort_key = "010",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/fonts/custom-font",
    .title = N_("Use Custom Font"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TERMINAL,
      .setting.key = "use-custom-font",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_FONT,
    .subpath = "/fonts/custom-font/font",
    .title = N_("Custom Font"),
    .flags = FOUNDRY_TWEAK_INFO_FONT_MONOSPACE,
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TERMINAL,
      .setting.key = "custom-font",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/styling",
    .sort_key = "020",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/styling/allow-bold",
    .title = N_("Allow Bold"),
    .subtitle = N_("Allow the use of bold escape sequences"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TERMINAL,
      .setting.key = "allow-bold",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/styling/allow-hyperlinks",
    .title = N_("Allow Hyperlinks"),
    .subtitle = N_("Allow the use of hyperlinks escape sequences"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TERMINAL,
      .setting.key = "allow-hyperlink",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/scrolling",
    .title = N_("Scrolling"),
    .sort_key = "030",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/scrolling/scroll-on-output",
    .title = N_("Scroll On Output"),
    .subtitle = N_("Automatically scroll when applications within the terminal output text"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TERMINAL,
      .setting.key = "scroll-on-output",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/scrolling/scroll-on-keystroke",
    .title = N_("Scroll On Keyboard Input"),
    .subtitle = N_("Automatically scroll when typing to insert text"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TERMINAL,
      .setting.key = "scroll-on-keystroke",
    },
  },

  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/history",
    .title = N_("History"),
    .sort_key = "040",
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SWITCH,
    .subpath = "/history/limit-scrollback",
    .title = N_("Limit Scrollback"),
    .subtitle = N_("Limit the number of lines that are stored in memory for terminal scrollback"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TERMINAL,
      .setting.key = "limit-scrollback",
    },
  },
  {
    .type = FOUNDRY_TWEAK_TYPE_SPIN,
    .subpath = "/history/max-scrollback-lines",
    .title = N_("Maximum Lines in Scrollback"),
    .subtitle = N_("The maximum number of lines stored in history when limiting scrollback"),
    .source = &(FoundryTweakSource) {
      .type = FOUNDRY_TWEAK_SOURCE_TYPE_SETTING,
      .setting.schema_id = APP_DEVSUITE_FOUNDRY_TERMINAL,
      .setting.key = "max-scrollback-lines",
    },
  },
};

static char *
find_icon_name (GtkSourceLanguage *language)
{
  g_auto(GStrv) mime_types = gtk_source_language_get_mime_types (language);
  const char *suffix = gtk_source_language_get_metadata (language, "suggested-suffix");
  g_autofree char *filename = NULL;

  if (suffix != NULL)
    filename = g_strdup_printf ("file%s", suffix);

  if (mime_types != NULL)
    {
      for (guint i = 0; mime_types[i]; i++)
        {
          g_autofree char *content_type = g_content_type_from_mime_type (mime_types[i]);

          if (!foundry_str_empty0 (content_type))
            {
              g_autoptr(GIcon) icon = foundry_file_manager_find_symbolic_icon (NULL, content_type, filename);

              if (icon != NULL)
                return g_icon_to_string (icon);
            }
        }
    }

  return g_strdup ("text-x-generic-symbolic");
}

static DexFuture *
foundry_gtk_tweak_provider_load (FoundryTweakProvider *provider)
{
  static const char *prefixes[] = {"/app", "/project", "/user"};
  GtkSourceLanguageManager *manager;
  const char * const *language_ids;
  g_auto(GStrv) basic_env = NULL;

  g_assert (FOUNDRY_IS_GTK_TWEAK_PROVIDER (provider));

  manager = gtk_source_language_manager_get_default ();
  language_ids = gtk_source_language_manager_get_language_ids (manager);

  basic_env = g_environ_setenv (basic_env, "language", "", TRUE);
  basic_env = g_environ_setenv (basic_env, "Language", "", TRUE);
  basic_env = g_environ_setenv (basic_env, "icon", "", TRUE);
  basic_env = g_environ_setenv (basic_env, "section", "", TRUE);

  foundry_tweak_provider_register (provider,
                                   GETTEXT_PACKAGE,
                                   "/app/terminal",
                                   terminal_infos,
                                   G_N_ELEMENTS (terminal_infos),
                                   NULL);

  foundry_tweak_provider_register (provider,
                                   GETTEXT_PACKAGE,
                                   "/app/editor",
                                   editor_infos,
                                   G_N_ELEMENTS (editor_infos),
                                   (const char * const *)basic_env);

  for (guint i = 0; i < G_N_ELEMENTS (prefixes); i++)
    {
      const char *prefix = prefixes[i];

      foundry_tweak_provider_register (provider,
                                       GETTEXT_PACKAGE,
                                       prefix,
                                       top_page_info,
                                       G_N_ELEMENTS (top_page_info),
                                       NULL);

      for (guint j = 0; language_ids[j]; j++)
        {
          const char *language_id = language_ids[j];
          GtkSourceLanguage *language = gtk_source_language_manager_get_language (manager, language_id);
          const char *name = gtk_source_language_get_name (language);
          g_autofree char *path = g_strdup_printf ("%s/editor/languages/%s/", prefix, language_id);
          g_autofree char *icon_name = NULL;
          g_auto(GStrv) environ_ = NULL;
          const char *section;

          if (gtk_source_language_get_hidden (language))
            continue;

          if (!(section = gtk_source_language_get_section (language)))
            section = "";

          icon_name = find_icon_name (language);

          environ_ = g_environ_setenv (environ_, "language", language_id, TRUE);
          environ_ = g_environ_setenv (environ_, "Language", name, TRUE);
          environ_ = g_environ_setenv (environ_, "icon", icon_name, TRUE);
          environ_ = g_environ_setenv (environ_, "section", section, TRUE);

          foundry_tweak_provider_register (provider,
                                           GETTEXT_PACKAGE,
                                           path,
                                           language_infos,
                                           G_N_ELEMENTS (language_infos),
                                           (const char * const *)environ_);

          foundry_tweak_provider_register (provider,
                                           GETTEXT_PACKAGE,
                                           path,
                                           editor_infos,
                                           G_N_ELEMENTS (editor_infos),
                                           (const char * const *)environ_);
        }
    }


  return dex_future_new_true ();
}

static void
foundry_gtk_tweak_provider_class_init (FoundryGtkTweakProviderClass *klass)
{
  FoundryTweakProviderClass *tweak_provider_class = FOUNDRY_TWEAK_PROVIDER_CLASS (klass);

  tweak_provider_class->load = foundry_gtk_tweak_provider_load;
}

static void
foundry_gtk_tweak_provider_init (FoundryGtkTweakProvider *self)
{
}
