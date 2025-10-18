/* foundry-service.h
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

#include "foundry-contextual.h"
#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_SERVICE  (foundry_service_get_type())
#define FOUNDRY_SERVICE_ERROR (foundry_service_error_quark())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryService, foundry_service, FOUNDRY, SERVICE, FoundryContextual)

typedef enum _FoundryServiceError
{
  FOUNDRY_SERVICE_ERROR_ALREADY_STARTED = 1,
  FOUNDRY_SERVICE_ERROR_ALREADY_STOPPED,
} FoundryServiceError;

struct _FoundryServiceClass
{
  FoundryContextualClass parent_class;

  DexFuture  *(*start) (FoundryService *self);
  DexFuture  *(*stop)  (FoundryService *self);

  /*< private >*/
  gpointer _reserved[8];
};

typedef void (*FoundryServiceAction) (FoundryService *self,
                                      const char     *action_name,
                                      GVariant       *param);

FOUNDRY_AVAILABLE_IN_ALL
GQuark      foundry_service_error_quark             (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
const char *foundry_service_class_get_action_prefix (FoundryServiceClass  *service_class);
FOUNDRY_AVAILABLE_IN_ALL
void        foundry_service_class_set_action_prefix (FoundryServiceClass  *service_class,
                                                     const char           *action_prefix);
FOUNDRY_AVAILABLE_IN_ALL
void        foundry_service_class_install_action    (FoundryServiceClass  *service_class,
                                                     const char           *action_name,
                                                     const char           *parameter_type,
                                                     FoundryServiceAction  activate);

FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_service_when_ready              (FoundryService       *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_service_when_shutdown           (FoundryService       *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
gboolean    foundry_service_action_get_enabled      (FoundryService       *self,
                                                     const char           *action);
FOUNDRY_AVAILABLE_IN_ALL
void        foundry_service_action_set_enabled      (FoundryService       *self,
                                                     const char           *action,
                                                     gboolean              enabled);

G_END_DECLS
