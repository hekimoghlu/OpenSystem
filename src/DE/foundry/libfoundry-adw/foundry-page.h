/* foundry-page.h
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

#include <libdex.h>
#include <gtk/gtk.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_PAGE (foundry_page_get_type())

FOUNDRY_AVAILABLE_IN_1_1
G_DECLARE_DERIVABLE_TYPE (FoundryPage, foundry_page, FOUNDRY, PAGE, GtkWidget)

struct _FoundryPageClass
{
  GtkWidgetClass parent_instance;

  gboolean    (*can_save)     (FoundryPage *self);
  char       *(*dup_title)    (FoundryPage *self);
  char       *(*dup_subtitle) (FoundryPage *self);
  GIcon      *(*dup_icon)     (FoundryPage *self);
  GMenuModel *(*dup_menu)     (FoundryPage *self);
  DexFuture  *(*save)         (FoundryPage *self);
  DexFuture  *(*save_as)      (FoundryPage *self);

  /*< private >*/
  gpointer _reserved[14];
};

FOUNDRY_AVAILABLE_IN_1_1
void       foundry_page_class_install_action          (FoundryPageClass            *page_class,
                                                       const char                  *action_name,
                                                       const char                  *parameter_type,
                                                       GtkWidgetActionActivateFunc  activate);
FOUNDRY_AVAILABLE_IN_1_1
void       foundry_page_class_install_property_action (FoundryPageClass            *page_class,
                                                       const char                  *action_name,
                                                       const char                  *property_name);
FOUNDRY_AVAILABLE_IN_1_1
gboolean    foundry_page_can_save                      (FoundryPage                 *self);
FOUNDRY_AVAILABLE_IN_1_1
char       *foundry_page_dup_title                     (FoundryPage                 *self);
FOUNDRY_AVAILABLE_IN_1_1
char       *foundry_page_dup_subtitle                  (FoundryPage                 *self);
FOUNDRY_AVAILABLE_IN_1_1
GIcon      *foundry_page_dup_icon                      (FoundryPage                 *self);
FOUNDRY_AVAILABLE_IN_1_1
GMenuModel *foundry_page_dup_menu                      (FoundryPage                 *self);
FOUNDRY_AVAILABLE_IN_1_1
GtkWidget  *foundry_page_get_content                   (FoundryPage                 *self);
FOUNDRY_AVAILABLE_IN_1_1
void        foundry_page_set_content                   (FoundryPage                 *self,
                                                        GtkWidget                   *content);
FOUNDRY_AVAILABLE_IN_1_1
GtkWidget  *foundry_page_get_auxillary                 (FoundryPage                 *self);
FOUNDRY_AVAILABLE_IN_1_1
void        foundry_page_set_auxillary                 (FoundryPage                 *self,
                                                        GtkWidget                   *auxillary);
FOUNDRY_AVAILABLE_IN_1_1
void        foundry_page_action_set_enabled            (FoundryPage                 *self,
                                                        const char                  *action_name,
                                                        gboolean                     enabled);
FOUNDRY_AVAILABLE_IN_1_1
void        foundry_page_raise                         (FoundryPage                 *self);
FOUNDRY_AVAILABLE_IN_1_1
DexFuture  *foundry_page_save                          (FoundryPage                 *self);
FOUNDRY_AVAILABLE_IN_1_1
DexFuture  *foundry_page_save_as                       (FoundryPage                 *self);

G_END_DECLS
