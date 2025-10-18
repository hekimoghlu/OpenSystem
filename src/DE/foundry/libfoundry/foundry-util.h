/* foundry-util.h
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

#pragma once

#include <errno.h>

#include <gobject/gvaluecollector.h>

#include <libdex.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_STRV_INIT(...) ((const char * const[]) { __VA_ARGS__, NULL})

FOUNDRY_AVAILABLE_IN_ALL
char       *foundry_dup_projects_directory      (void);
FOUNDRY_AVAILABLE_IN_ALL
GFile      *foundry_dup_projects_directory_file (void);
FOUNDRY_AVAILABLE_IN_ALL
const char *foundry_get_default_arch            (void);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_key_file_new_merged         (const char * const  *search_dirs,
                                                 const char          *file,
                                                 GKeyFileFlags        flags) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_key_file_new_from_file      (GFile               *file,
                                                 GKeyFileFlags        flags) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_file_test                   (const char          *path,
                                                 GFileTest            test) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
const char *foundry_get_version_string          (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
gboolean    foundry_pipe                        (int                 *read_fd,
                                                 int                 *write_fd,
                                                 int                  flags,
                                                 GError             **error);

#if defined(_MSC_VER)
# define FOUNDRY_ALIGNED_BEGIN(_N) __declspec(align(_N))
# define FOUNDRY_ALIGNED_END(_N)
#else
# define FOUNDRY_ALIGNED_BEGIN(_N)
# define FOUNDRY_ALIGNED_END(_N) __attribute__((aligned(_N)))
#endif

#ifndef __GI_SCANNER__

typedef struct _FoundryPair
{
  GObject *first;
  GObject *second;
} FoundryPair;

static inline FoundryPair *
foundry_pair_new (gpointer first,
                  gpointer second)
{
  FoundryPair *pair = g_new0 (FoundryPair, 1);
  g_set_object (&pair->first, first);
  g_set_object (&pair->second, second);
  return pair;
}

static inline void
foundry_pair_free (FoundryPair *pair)
{
  g_clear_object (&pair->first);
  g_clear_object (&pair->second);
  g_free (pair);
}

static inline gboolean
foundry_set_strv (char ***ptr,
                  const char * const *strv)
{
  char **copy;

  if ((const char * const *)*ptr == strv)
    return FALSE;

  if (*ptr && strv && g_strv_equal ((const char * const *)*ptr, strv))
    return FALSE;

  copy = g_strdupv ((char **)strv);
  g_strfreev (*ptr);
  *ptr = copy;

  return TRUE;
}

G_GNUC_WARN_UNUSED_RESULT
static inline char **
foundry_strv_append (char       **strv,
                     const char  *str)
{
  gsize len = strv ? g_strv_length (strv) : 0;

  if (strv == NULL)
    strv = g_new0 (char *, 2);
  else
    strv = g_realloc_n (strv, len + 2, sizeof (char *));

  strv[len++] = g_strdup (str);
  strv[len] = NULL;

  return strv;
}

static inline void
foundry_take_str (char **strptr,
                  char *str)
{
  if (g_strcmp0 (*strptr, str) == 0)
    {
      g_free (str);
    }
  else
    {
      g_free (*strptr);
      *strptr = str;
    }
}

static inline gboolean
foundry_str_equal0 (const char *a,
                    const char *b)
{
  return a == b || g_strcmp0 (a, b) == 0;
}

static inline gboolean
foundry_str_empty0 (const char *str)
{
  return str == NULL || str[0] == 0;
}

G_GNUC_WARN_UNUSED_RESULT
static inline DexFuture *
foundry_future_all (GPtrArray *ar)
{
  g_assert (ar != NULL);
  g_assert (ar->len > 0);

  return dex_future_allv ((DexFuture **)ar->pdata, ar->len);
}

static inline DexFuture *
foundry_future_return_object (DexFuture *future,
                              gpointer   user_data)
{
  return dex_future_new_take_object (g_object_ref (user_data));
}

static inline DexFuture *
foundry_future_return_true (DexFuture *future,
                            gpointer   user_data)
{
  return dex_future_new_true ();
}

static inline gsize
foundry_set_error_from_errno (GError **error)
{
  int errsv = errno;
  if (error != NULL)
    g_set_error_literal (error,
                         G_IO_ERROR,
                         g_io_error_from_errno (errsv),
                         g_strerror (errsv));
  return 0;
}

static inline DexFuture *
foundry_future_new_disposed (void)
{
  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_FAILED,
                                "Object disposed");
}

static inline DexFuture *
foundry_future_new_not_supported (void)
{
  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "Not supported");
}

typedef struct _FoundryTrampoline
{
  GCallback callback;
  GArray *values;
} FoundryTrampoline;

static void
foundry_trampoline_free (FoundryTrampoline *state)
{
  state->callback = NULL;
  g_clear_pointer (&state->values, g_array_unref);
  g_free (state);
}

static inline DexFuture *
foundry_trampoline_fiber (gpointer data)
{
  FoundryTrampoline *state = data;
  g_autoptr(GClosure) closure = NULL;
  g_auto(GValue) return_value = G_VALUE_INIT;

  g_assert (state != NULL);
  g_assert (state->callback != NULL);
  g_assert (state->values != NULL);

  g_value_init (&return_value, G_TYPE_POINTER);
  closure = g_cclosure_new (state->callback, NULL, NULL);
  g_closure_set_marshal (closure, g_cclosure_marshal_generic);
  g_closure_invoke (closure,
                    &return_value,
                    state->values->len,
                    (const GValue *)(gpointer)state->values->data,
                    NULL);
  return g_value_get_pointer (&return_value);
}

/**
 * foundry_scheduler_spawn:
 *
 * Trampoline into a fiber without having to create
 * special structures on your way there.
 *
 * @n_params should denote the number of pairs of
 * #GType followed by value to collect from the va_list.
 */
static inline DexFuture *
foundry_scheduler_spawn (DexScheduler *scheduler,
                         gsize         stack_size,
                         GCallback     callback,
                         guint         n_params,
                         ...)
{
  g_autofree char *errmsg = NULL;
  g_autoptr(GArray) values = NULL;
  FoundryTrampoline *state;
  va_list args;

  values = g_array_new (FALSE, TRUE, sizeof (GValue));
  g_array_set_clear_func (values, (GDestroyNotify)g_value_unset);
  g_array_set_size (values, n_params);

  va_start (args, n_params);

  for (guint i = 0; i < n_params; i++)
    {
      GType gtype = va_arg (args, GType);
      GValue *dest = &g_array_index (values, GValue, i);
      g_auto(GValue) value = G_VALUE_INIT;

      G_VALUE_COLLECT_INIT (&value, gtype, args, 0, &errmsg);

      if (errmsg != NULL)
        break;

      g_value_init (dest, gtype);
      g_value_copy (&value, dest);
    }

  va_end (args);

  if (errmsg != NULL)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_FAILED,
                                  "Failed to trampoline to fiber: %s",
                                  errmsg);

  state = g_new0 (FoundryTrampoline, 1);
  state->values = g_steal_pointer (&values);
  state->callback = callback;

  return dex_scheduler_spawn (scheduler, stack_size,
                              foundry_trampoline_fiber,
                              state,
                              (GDestroyNotify) foundry_trampoline_free);
}

#endif

G_END_DECLS
