/* foundry-contextual.h
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

#include <glib-object.h>

#include "foundry-context.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_CONTEXTUAL  (foundry_contextual_get_type())
#define FOUNDRY_CONTEXTUAL_ERROR (foundry_contextual_error_quark())

typedef enum _FoundryContextualError
{
  FOUNDRY_CONTEXTUAL_ERROR_IN_SHUTDOWN = 1,
} FoundryContextualError;

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryContextual, foundry_contextual, FOUNDRY, CONTEXTUAL, GObject)

struct _FoundryContextualClass
{
  GObjectClass parent_class;

  const char *log_domain;

  /*< private >*/
  gpointer _reserved[6];
};

FOUNDRY_AVAILABLE_IN_ALL
GQuark            foundry_contextual_error_quark (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
FoundryInhibitor *foundry_contextual_inhibit     (FoundryContextual  *self,
                                                  GError            **error);
FOUNDRY_AVAILABLE_IN_ALL
FoundryContext   *foundry_contextual_dup_context (FoundryContextual  *self);
FOUNDRY_AVAILABLE_IN_ALL
void              foundry_contextual_log         (FoundryContextual  *self,
                                                  const char         *domain,
                                                  GLogLevelFlags      severity,
                                                  const char         *format,
                                                  ...) G_GNUC_PRINTF (4, 5);


#define FOUNDRY_CONTEXTUAL_DEBUG(contextual, format, ...) \
  foundry_contextual_log(FOUNDRY_CONTEXTUAL (contextual), G_LOG_DOMAIN, G_LOG_LEVEL_DEBUG, format, __VA_ARGS__)
#define FOUNDRY_CONTEXTUAL_MESSAGE(contextual, format, ...) \
  foundry_contextual_log(FOUNDRY_CONTEXTUAL (contextual), G_LOG_DOMAIN, G_LOG_LEVEL_MESSAGE, format, __VA_ARGS__)
#define FOUNDRY_CONTEXTUAL_WARNING(contextual, format, ...) \
  foundry_contextual_log(FOUNDRY_CONTEXTUAL (contextual), G_LOG_DOMAIN, G_LOG_LEVEL_WARNING, format, __VA_ARGS__)

G_END_DECLS
