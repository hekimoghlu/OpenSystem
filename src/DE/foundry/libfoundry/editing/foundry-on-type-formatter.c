/* foundry-on-type-formatter.c
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

#include "foundry-text-document.h"
#include "foundry-on-type-formatter.h"

G_DEFINE_ABSTRACT_TYPE (FoundryOnTypeFormatter, foundry_on_type_formatter, G_TYPE_OBJECT)

static void
foundry_on_type_formatter_class_init (FoundryOnTypeFormatterClass *klass)
{
}

static void
foundry_on_type_formatter_init (FoundryOnTypeFormatter *self)
{
}

gboolean
foundry_on_type_formatter_is_trigger (FoundryOnTypeFormatter *self,
                                      FoundryTextDocument    *document,
                                      const FoundryTextIter  *iter,
                                      FoundryModifierType     state,
                                      guint                   keyval)
{
  g_return_val_if_fail (FOUNDRY_IS_ON_TYPE_FORMATTER (self), FALSE);
  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (document), FALSE);
  g_return_val_if_fail (iter != NULL, FALSE);

  return FOUNDRY_ON_TYPE_FORMATTER_GET_CLASS (self)->is_trigger (self, document, iter, state, keyval);
}

/**
 * foundry_on_type_formatter_indent:
 * @self: a [class@Foundry.OnTypeFormatter]
 * @document:
 * @iter: (inout):
 *
 * Indents the text at @iter.
 *
 * @iter should be set to the cursor location after the indent
 * when exiting this function.
 */
void
foundry_on_type_formatter_indent (FoundryOnTypeFormatter *self,
                                  FoundryTextDocument    *document,
                                  FoundryTextIter        *iter)
{
  g_return_if_fail (FOUNDRY_IS_ON_TYPE_FORMATTER (self));
  g_return_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (document));
  g_return_if_fail (iter != NULL);

  FOUNDRY_ON_TYPE_FORMATTER_GET_CLASS (self)->indent (self, document, iter);
}

G_DEFINE_FLAGS_TYPE (FoundryModifierType, foundry_modifier_type,
                     G_DEFINE_ENUM_VALUE (FOUNDRY_MODIFIER_CONTROL, "control"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_MODIFIER_SHIFT, "shift"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_MODIFIER_ALT, "alt"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_MODIFIER_SUPER, "super"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_MODIFIER_COMMAND, "command"))

