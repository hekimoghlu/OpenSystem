/* foundry-operation.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include "foundry-auth-provider.h"
#include "foundry-operation.h"
#include "foundry-util-private.h"

struct _FoundryOperation
{
  GObject              parent_instance;
  FoundryAuthProvider *auth_provider;
  DexPromise          *completion;
  GMutex               mutex;
  char                *title;
  char                *subtitle;
  double               progress;
};

enum {
  PROP_0,
  PROP_AUTH_PROVIDER,
  PROP_PROGRESS,
  PROP_SUBTITLE,
  PROP_TITLE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryOperation, foundry_operation, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_operation_dispose (GObject *object)
{
  FoundryOperation *self = (FoundryOperation *)object;

  foundry_operation_cancel (self);

  G_OBJECT_CLASS (foundry_operation_parent_class)->dispose (object);
}

static void
foundry_operation_finalize (GObject *object)
{
  FoundryOperation *self = (FoundryOperation *)object;

  g_mutex_clear (&self->mutex);
  g_clear_pointer (&self->title, g_free);
  g_clear_pointer (&self->subtitle, g_free);
  g_clear_object (&self->auth_provider);
  dex_clear (&self->completion);

  G_OBJECT_CLASS (foundry_operation_parent_class)->finalize (object);
}

static void
foundry_operation_get_property (GObject    *object,
                                guint       prop_id,
                                GValue     *value,
                                GParamSpec *pspec)
{
  FoundryOperation *self = FOUNDRY_OPERATION (object);

  switch (prop_id)
    {
    case PROP_AUTH_PROVIDER:
      g_value_take_object (value, foundry_operation_dup_auth_provider (self));
      break;

    case PROP_PROGRESS:
      g_value_set_double (value, foundry_operation_get_progress (self));
      break;

    case PROP_SUBTITLE:
      g_value_take_string (value, foundry_operation_dup_subtitle (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, foundry_operation_dup_title (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_operation_set_property (GObject      *object,
                                guint         prop_id,
                                const GValue *value,
                                GParamSpec   *pspec)
{
  FoundryOperation *self = FOUNDRY_OPERATION (object);

  switch (prop_id)
    {
    case PROP_AUTH_PROVIDER:
      foundry_operation_set_auth_provider (self, g_value_get_object (value));
      break;

    case PROP_PROGRESS:
      foundry_operation_set_progress (self, g_value_get_double (value));
      break;

    case PROP_SUBTITLE:
      foundry_operation_set_subtitle (self, g_value_get_string (value));
      break;

    case PROP_TITLE:
      foundry_operation_set_title (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_operation_class_init (FoundryOperationClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_operation_dispose;
  object_class->finalize = foundry_operation_finalize;
  object_class->get_property = foundry_operation_get_property;
  object_class->set_property = foundry_operation_set_property;

  properties[PROP_AUTH_PROVIDER] =
    g_param_spec_object ("auth-provider", NULL, NULL,
                         FOUNDRY_TYPE_AUTH_PROVIDER,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PROGRESS] =
    g_param_spec_double ("progress", NULL, NULL,
                         0, 1, 0,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SUBTITLE] =
    g_param_spec_string ("subtitle", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_operation_init (FoundryOperation *self)
{
  g_mutex_init (&self->mutex);
  self->completion = dex_promise_new ();
}

double
foundry_operation_get_progress (FoundryOperation *self)
{
  g_autoptr(GMutexLocker) locker = NULL;

  g_return_val_if_fail (FOUNDRY_IS_OPERATION (self), 0);

  locker = g_mutex_locker_new (&self->mutex);
  return self->progress;
}

void
foundry_operation_set_progress (FoundryOperation *self,
                                double            progress)
{
  gboolean changed = FALSE;

  g_return_if_fail (FOUNDRY_IS_OPERATION (self));

  g_mutex_lock (&self->mutex);
  if (progress != self->progress)
    {
      self->progress = progress;
      changed = TRUE;
    }
  g_mutex_unlock (&self->mutex);

  if (changed)
    foundry_notify_pspec_in_main (G_OBJECT (self), properties[PROP_PROGRESS]);
}

char *
foundry_operation_dup_subtitle (FoundryOperation *self)
{
  g_autoptr(GMutexLocker) locker = NULL;

  g_return_val_if_fail (FOUNDRY_IS_OPERATION (self), NULL);

  locker = g_mutex_locker_new (&self->mutex);
  return g_strdup (self->subtitle);
}

void
foundry_operation_set_subtitle (FoundryOperation *self,
                                const char       *subtitle)
{
  gboolean changed;

  g_return_if_fail (FOUNDRY_IS_OPERATION (self));

  g_mutex_lock (&self->mutex);
  changed = g_set_str (&self->subtitle, subtitle);
  g_mutex_unlock (&self->mutex);

  if (changed)
    foundry_notify_pspec_in_main (G_OBJECT (self), properties[PROP_SUBTITLE]);
}

char *
foundry_operation_dup_title (FoundryOperation *self)
{
  g_autoptr(GMutexLocker) locker = NULL;

  g_return_val_if_fail (FOUNDRY_IS_OPERATION (self), NULL);

  locker = g_mutex_locker_new (&self->mutex);
  return g_strdup (self->title);
}

void
foundry_operation_set_title (FoundryOperation *self,
                             const char       *title)
{
  gboolean changed;

  g_return_if_fail (FOUNDRY_IS_OPERATION (self));

  g_mutex_lock (&self->mutex);
  changed = g_set_str (&self->title, title);
  g_mutex_unlock (&self->mutex);

  if (changed)
    foundry_notify_pspec_in_main (G_OBJECT (self), properties[PROP_TITLE]);
}

void
foundry_operation_cancel (FoundryOperation *self)
{
  g_return_if_fail (FOUNDRY_IS_OPERATION (self));

  if (dex_future_is_pending (DEX_FUTURE (self->completion)))
    dex_promise_reject (self->completion,
                        g_error_new (G_IO_ERROR,
                                     G_IO_ERROR_CANCELLED,
                                     "Operation cancelled"));
}

void
foundry_operation_complete (FoundryOperation *self)
{
  g_return_if_fail (FOUNDRY_IS_OPERATION (self));

  if (dex_future_is_pending (DEX_FUTURE (self->completion)))
    dex_promise_resolve_object (self->completion, g_object_ref (self));
}

/**
 * foundry_operation_await:
 * @self: a #FoundryOperation
 *
 * Gets a [class@Dex.Future] that resolves when the operation
 * has cancelled or completed.
 *
 * Returns: (transfer full): a [class@Dex.Future].
 */
DexFuture *
foundry_operation_await (FoundryOperation *self)
{
  g_return_val_if_fail (FOUNDRY_IS_OPERATION (self), NULL);

  return dex_ref (self->completion);
}

void
foundry_operation_file_progress (goffset  current_num_bytes,
                                 goffset  total_num_bytes,
                                 gpointer user_data)
{
  FoundryOperation *self = user_data;
  gboolean progress;

  g_return_if_fail (FOUNDRY_IS_OPERATION (self));

  if (total_num_bytes == 0)
    progress = 0;
  else
    progress = CLAMP ((double)current_num_bytes / (double)total_num_bytes, 0., 1.);

  foundry_operation_set_progress (self, progress);
}

FoundryOperation *
foundry_operation_new (void)
{
  return g_object_new (FOUNDRY_TYPE_OPERATION, NULL);
}

/**
 * foundry_operation_dup_auth_provider:
 * @self: a [class@Foundry.Operation]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryAuthProvider *
foundry_operation_dup_auth_provider (FoundryOperation *self)
{
  g_autoptr(GMutexLocker) locker = NULL;

  g_return_val_if_fail (FOUNDRY_IS_OPERATION (self), NULL);

  locker = g_mutex_locker_new (&self->mutex);

  return self->auth_provider ? g_object_ref (self->auth_provider) : NULL;
}

void
foundry_operation_set_auth_provider (FoundryOperation    *self,
                                     FoundryAuthProvider *auth_provider)
{
  gboolean changed;

  g_return_if_fail (FOUNDRY_IS_OPERATION (self));
  g_return_if_fail (!auth_provider || FOUNDRY_IS_AUTH_PROVIDER (auth_provider));

  g_mutex_lock (&self->mutex);
  changed = g_set_object (&self->auth_provider, auth_provider);
  g_mutex_unlock (&self->mutex);

  if (changed)
    foundry_notify_pspec_in_main (G_OBJECT (self), properties[PROP_AUTH_PROVIDER]);
}
