/* foundry-text-document-addin.c
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
#include "foundry-text-document-addin-private.h"
#include "foundry-util.h"

typedef struct
{
  GWeakRef document_wr;
} FoundryTextDocumentAddinPrivate;

enum {
  PROP_0,
  PROP_DOCUMENT,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryTextDocumentAddin, foundry_text_document_addin, FOUNDRY_TYPE_CONTEXTUAL)

static GParamSpec *properties[N_PROPS];

void
_foundry_text_document_addin_set_document (FoundryTextDocumentAddin *self,
                                           FoundryTextDocument      *document)
{
  FoundryTextDocumentAddinPrivate *priv = foundry_text_document_addin_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_TEXT_DOCUMENT_ADDIN (self));
  g_return_if_fail (!document || FOUNDRY_IS_TEXT_DOCUMENT (document));

  g_weak_ref_set (&priv->document_wr, document);
}

static void
foundry_text_document_addin_finalize (GObject *object)
{
  FoundryTextDocumentAddin *self = (FoundryTextDocumentAddin *)object;
  FoundryTextDocumentAddinPrivate *priv = foundry_text_document_addin_get_instance_private (self);

  g_weak_ref_clear (&priv->document_wr);

  G_OBJECT_CLASS (foundry_text_document_addin_parent_class)->finalize (object);
}

static void
foundry_text_document_addin_get_property (GObject    *object,
                                          guint       prop_id,
                                          GValue     *value,
                                          GParamSpec *pspec)
{
  FoundryTextDocumentAddin *self = FOUNDRY_TEXT_DOCUMENT_ADDIN (object);

  switch (prop_id)
    {
    case PROP_DOCUMENT:
      g_value_take_object (value, foundry_text_document_addin_dup_document (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_text_document_addin_set_property (GObject      *object,
                                          guint         prop_id,
                                          const GValue *value,
                                          GParamSpec   *pspec)
{
  FoundryTextDocumentAddin *self = FOUNDRY_TEXT_DOCUMENT_ADDIN (object);
  FoundryTextDocumentAddinPrivate *priv = foundry_text_document_addin_get_instance_private (self);

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
foundry_text_document_addin_class_init (FoundryTextDocumentAddinClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_text_document_addin_finalize;
  object_class->get_property = foundry_text_document_addin_get_property;
  object_class->set_property = foundry_text_document_addin_set_property;

  properties[PROP_DOCUMENT] =
    g_param_spec_object ("document", NULL, NULL,
                         FOUNDRY_TYPE_TEXT_DOCUMENT,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_text_document_addin_init (FoundryTextDocumentAddin *self)
{
  FoundryTextDocumentAddinPrivate *priv = foundry_text_document_addin_get_instance_private (self);

  g_weak_ref_init (&priv->document_wr, NULL);
}

/**
 * foundry_text_document_addin_dup_document:
 * @self: a [class@Foundry.TextDocumentAddin]
 *
 * Returns: (transfer full):
 */
FoundryTextDocument *
foundry_text_document_addin_dup_document (FoundryTextDocumentAddin *self)
{
  FoundryTextDocumentAddinPrivate *priv = foundry_text_document_addin_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT_ADDIN (self), NULL);

  return g_weak_ref_get (&priv->document_wr);
}

/**
 * foundry_text_document_addin_load:
 * @self: a [class@Foundry.TextDocumentAddin]
 *
 * Returns: (transfer full):
 */
DexFuture *
foundry_text_document_addin_load (FoundryTextDocumentAddin *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT_ADDIN (self), NULL);

  if (FOUNDRY_TEXT_DOCUMENT_ADDIN_GET_CLASS (self)->load)
    return FOUNDRY_TEXT_DOCUMENT_ADDIN_GET_CLASS (self)->load (self);

  return dex_future_new_true ();
}

/**
 * foundry_text_document_addin_unload:
 * @self: a [class@Foundry.TextDocumentAddin]
 *
 * Returns: (transfer full):
 */
DexFuture *
foundry_text_document_addin_unload (FoundryTextDocumentAddin *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT_ADDIN (self), NULL);

  if (FOUNDRY_TEXT_DOCUMENT_ADDIN_GET_CLASS (self)->unload)
    return FOUNDRY_TEXT_DOCUMENT_ADDIN_GET_CLASS (self)->unload (self);

  return dex_future_new_true ();
}

/**
 * foundry_text_document_addin_pre_load:
 * @self: a [class@Foundry.TextDocumentAddin]
 *
 * Called before the document is loadd.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any
 *   value or rejects with error.
 */
DexFuture *
foundry_text_document_addin_pre_load (FoundryTextDocumentAddin *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_DOCUMENT_ADDIN (self));

  if (FOUNDRY_TEXT_DOCUMENT_ADDIN_GET_CLASS (self)->pre_load)
    return FOUNDRY_TEXT_DOCUMENT_ADDIN_GET_CLASS (self)->pre_load (self);

  return dex_future_new_true ();
}

/**
 * foundry_text_document_addin_post_load:
 * @self: a [class@Foundry.TextDocumentAddin]
 *
 * Called after the document is loadd.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any
 *   value or rejects with error.
 */
DexFuture *
foundry_text_document_addin_post_load (FoundryTextDocumentAddin *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_DOCUMENT_ADDIN (self));

  if (FOUNDRY_TEXT_DOCUMENT_ADDIN_GET_CLASS (self)->post_load)
    return FOUNDRY_TEXT_DOCUMENT_ADDIN_GET_CLASS (self)->post_load (self);

  return dex_future_new_true ();
}

/**
 * foundry_text_document_addin_pre_save:
 * @self: a [class@Foundry.TextDocumentAddin]
 *
 * Called before the document is saved.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any
 *   value or rejects with error.
 */
DexFuture *
foundry_text_document_addin_pre_save (FoundryTextDocumentAddin *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_DOCUMENT_ADDIN (self));

  if (FOUNDRY_TEXT_DOCUMENT_ADDIN_GET_CLASS (self)->pre_save)
    return FOUNDRY_TEXT_DOCUMENT_ADDIN_GET_CLASS (self)->pre_save (self);

  return dex_future_new_true ();
}

/**
 * foundry_text_document_addin_post_save:
 * @self: a [class@Foundry.TextDocumentAddin]
 *
 * Called after the document is saved.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any
 *   value or rejects with error.
 */
DexFuture *
foundry_text_document_addin_post_save (FoundryTextDocumentAddin *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_DOCUMENT_ADDIN (self));

  if (FOUNDRY_TEXT_DOCUMENT_ADDIN_GET_CLASS (self)->post_save)
    return FOUNDRY_TEXT_DOCUMENT_ADDIN_GET_CLASS (self)->post_save (self);

  return dex_future_new_true ();
}

/**
 * foundry_text_document_addin_list_code_actions:
 * @self: a [class@Foundry.TextDocumentAddin]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.CodeAction] or rejects with error
 */
DexFuture *
foundry_text_document_addin_list_code_actions (FoundryTextDocumentAddin *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT_ADDIN (self), NULL);

  if (FOUNDRY_TEXT_DOCUMENT_ADDIN_GET_CLASS (self)->list_code_actions)
    return FOUNDRY_TEXT_DOCUMENT_ADDIN_GET_CLASS (self)->list_code_actions (self);

  return foundry_future_new_not_supported ();
}
