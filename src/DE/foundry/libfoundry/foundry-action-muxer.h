/* foundry-action-muxer.h
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

#pragma once

#include <gio/gio.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

typedef void (*FoundryActionActivateFunc) (gpointer    instance,
                                           const char *action_name,
                                           GVariant   *param);

typedef struct _FoundryAction
{
  const struct _FoundryAction *next;
  const char                  *name;
  GType                        owner;
  const GVariantType          *parameter_type;
  const GVariantType          *state_type;
  GParamSpec                  *pspec;
  FoundryActionActivateFunc    activate;
  guint                        position;
} FoundryAction;

typedef struct _FoundryActionMixin
{
  GObjectClass        *object_class;
  const FoundryAction *actions;
  guint                n_actions;
} FoundryActionMixin;

#define FOUNDRY_TYPE_ACTION_MUXER (foundry_action_muxer_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryActionMuxer, foundry_action_muxer, FOUNDRY, ACTION_MUXER, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryActionMuxer  *foundry_action_mixin_get_action_muxer        (gpointer                   instance);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_action_mixin_init                    (FoundryActionMixin        *mixin,
                                                                   GObjectClass              *object_class);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_action_mixin_constructed             (const FoundryActionMixin  *mixin,
                                                                   gpointer                   instance);
FOUNDRY_AVAILABLE_IN_ALL
gboolean             foundry_action_mixin_get_enabled             (gpointer                   instance,
                                                                   const char                *action);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_action_mixin_set_enabled             (gpointer                   instance,
                                                                   const char                *action,
                                                                   gboolean                   enabled);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_action_mixin_install_action          (FoundryActionMixin        *mixin,
                                                                   const char                *action_name,
                                                                   const char                *parameter_type,
                                                                   FoundryActionActivateFunc  activate);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_action_mixin_install_property_action (FoundryActionMixin        *mixin,
                                                                   const char                *action_name,
                                                                   const char                *property_name);
FOUNDRY_AVAILABLE_IN_ALL
FoundryActionMuxer  *foundry_action_muxer_new                     (void);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_action_muxer_remove_all              (FoundryActionMuxer        *self);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_action_muxer_insert_action_group     (FoundryActionMuxer        *self,
                                                                   const char                *prefix,
                                                                   GActionGroup              *action_group);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_action_muxer_remove_action_group     (FoundryActionMuxer        *self,
                                                                   const char                *prefix);
FOUNDRY_AVAILABLE_IN_ALL
char               **foundry_action_muxer_list_groups             (FoundryActionMuxer        *self);
FOUNDRY_AVAILABLE_IN_ALL
GActionGroup        *foundry_action_muxer_get_action_group        (FoundryActionMuxer        *self,
                                                                   const char                *prefix);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_action_muxer_set_enabled             (FoundryActionMuxer        *self,
                                                                   const FoundryAction       *action,
                                                                   gboolean                   enabled);
FOUNDRY_AVAILABLE_IN_ALL
gboolean             foundry_action_muxer_get_enabled             (FoundryActionMuxer        *self,
                                                                   const FoundryAction       *action);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_action_muxer_connect_actions         (FoundryActionMuxer        *self,
                                                                   gpointer                   instance,
                                                                   const FoundryAction       *actions);

G_END_DECLS
