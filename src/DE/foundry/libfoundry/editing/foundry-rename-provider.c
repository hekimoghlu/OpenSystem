/* foundry-rename-provider.c
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

#include "foundry-rename-provider.h"
#include "foundry-text-buffer.h"
#include "foundry-text-document.h"

typedef struct
{
  GWeakRef document_wr;
} FoundryRenameProviderPrivate;

enum {
  PROP_0,
  PROP_BUFFER,
  PROP_DOCUMENT,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryRenameProvider, foundry_rename_provider, FOUNDRY_TYPE_CONTEXTUAL)

static GParamSpec *properties[N_PROPS];

static void
foundry_rename_provider_dispose (GObject *object)
{
  FoundryRenameProvider *self = (FoundryRenameProvider *)object;
  FoundryRenameProviderPrivate *priv = foundry_rename_provider_get_instance_private (self);

  g_weak_ref_set (&priv->document_wr, NULL);

  G_OBJECT_CLASS (foundry_rename_provider_parent_class)->dispose (object);
}

static void
foundry_rename_provider_finalize (GObject *object)
{
  FoundryRenameProvider *self = (FoundryRenameProvider *)object;
  FoundryRenameProviderPrivate *priv = foundry_rename_provider_get_instance_private (self);

  g_weak_ref_clear (&priv->document_wr);

  G_OBJECT_CLASS (foundry_rename_provider_parent_class)->finalize (object);
}

static void
foundry_rename_provider_get_property (GObject    *object,
                                      guint       prop_id,
                                      GValue     *value,
                                      GParamSpec *pspec)
{
  FoundryRenameProvider *self = FOUNDRY_RENAME_PROVIDER (object);

  switch (prop_id)
    {
    case PROP_BUFFER:
      g_value_take_object (value, foundry_rename_provider_dup_buffer (self));
      break;

    case PROP_DOCUMENT:
      g_value_take_object (value, foundry_rename_provider_dup_document (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_rename_provider_set_property (GObject      *object,
                                      guint         prop_id,
                                      const GValue *value,
                                      GParamSpec   *pspec)
{
  FoundryRenameProvider *self = FOUNDRY_RENAME_PROVIDER (object);
  FoundryRenameProviderPrivate *priv = foundry_rename_provider_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_DOCUMENT:
      g_weak_ref_set (&priv->document_wr, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_rename_provider_class_init (FoundryRenameProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_rename_provider_dispose;
  object_class->finalize = foundry_rename_provider_finalize;
  object_class->get_property = foundry_rename_provider_get_property;
  object_class->set_property = foundry_rename_provider_set_property;

  properties[PROP_BUFFER] =
    g_param_spec_object ("buffer", NULL, NULL,
                         FOUNDRY_TYPE_TEXT_BUFFER,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_DOCUMENT] =
    g_param_spec_object ("document", NULL, NULL,
                         FOUNDRY_TYPE_TEXT_DOCUMENT,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_rename_provider_init (FoundryRenameProvider *self)
{
}

/**
 * foundry_rename_provider_dup_document:
 * @self: a [class@Foundry.RenameProvider]
 *
 * Returns: (transfer full):
 */
FoundryTextDocument *
foundry_rename_provider_dup_document (FoundryRenameProvider *self)
{
  FoundryRenameProviderPrivate *priv = foundry_rename_provider_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_RENAME_PROVIDER (self), NULL);

  return g_weak_ref_get (&priv->document_wr);
}

/**
 * foundry_rename_provider_dup_buffer:
 * @self: a [class@Foundry.RenameProvider]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryTextBuffer *
foundry_rename_provider_dup_buffer (FoundryRenameProvider *self)
{
  g_autoptr(FoundryTextDocument) document = NULL;

  g_return_val_if_fail (FOUNDRY_IS_RENAME_PROVIDER (self), NULL);

  if ((document = foundry_rename_provider_dup_document (self)))
    return foundry_text_document_dup_buffer (document);

  return NULL;
}

/**
 * foundry_rename_provider_rename:
 * @self: a [class@Foundry.RenameProvider]
 * @iter: the location of the item to semantically rename
 * @new_name: the replacement name
 *
 * Determines the list of changes that need to be made to the code-base
 * to rename the word found at @iter.
 *
 * A consuming interface should display these edits to the user for
 * validation and approval before applying them using
 * [method@Foundry.TextManager.apply_edits] to apply the approved
 * edits.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.TextEdit].
 */
DexFuture *
foundry_rename_provider_rename (FoundryRenameProvider *self,
                                const FoundryTextIter *iter,
                                const char            *new_name)
{
  dex_return_error_if_fail (FOUNDRY_IS_RENAME_PROVIDER (self));
  dex_return_error_if_fail (iter != NULL);
  dex_return_error_if_fail (new_name != NULL);

  if (FOUNDRY_RENAME_PROVIDER_GET_CLASS (self)->rename)
    return FOUNDRY_RENAME_PROVIDER_GET_CLASS (self)->rename (self, iter, new_name);

  return NULL;
}
