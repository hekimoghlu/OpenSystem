/* foundry-diagnostic-builder.h
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
#include "foundry-diagnostic.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DIAGNOSTIC_BUILDER (foundry_diagnostic_builder_get_type())

typedef struct _FoundryDiagnosticBuilder FoundryDiagnosticBuilder;

FOUNDRY_AVAILABLE_IN_ALL
GType                     foundry_diagnostic_builder_get_type        (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
FoundryDiagnosticBuilder *foundry_diagnostic_builder_new             (FoundryContext            *context);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDiagnosticBuilder *foundry_diagnostic_builder_ref             (FoundryDiagnosticBuilder  *self);
FOUNDRY_AVAILABLE_IN_ALL
void                      foundry_diagnostic_builder_unref           (FoundryDiagnosticBuilder  *self);
FOUNDRY_AVAILABLE_IN_1_1
void                      foundry_diagnostic_builder_set_rule_id     (FoundryDiagnosticBuilder  *self,
                                                                      const char                *rule_id);
FOUNDRY_AVAILABLE_IN_ALL
void                      foundry_diagnostic_builder_set_markup      (FoundryDiagnosticBuilder  *self,
                                                                      FoundryMarkup             *markup);
FOUNDRY_AVAILABLE_IN_ALL
void                      foundry_diagnostic_builder_take_markup     (FoundryDiagnosticBuilder  *self,
                                                                      FoundryMarkup             *markup);
FOUNDRY_AVAILABLE_IN_ALL
void                      foundry_diagnostic_builder_set_message     (FoundryDiagnosticBuilder  *self,
                                                                      const char                *text);
FOUNDRY_AVAILABLE_IN_ALL
void                      foundry_diagnostic_builder_take_message    (FoundryDiagnosticBuilder  *self,
                                                                      char                      *text);
FOUNDRY_AVAILABLE_IN_ALL
void                      foundry_diagnostic_builder_set_severity    (FoundryDiagnosticBuilder  *self,
                                                                      FoundryDiagnosticSeverity  severity);
FOUNDRY_AVAILABLE_IN_ALL
void                      foundry_diagnostic_builder_set_file        (FoundryDiagnosticBuilder  *self,
                                                                      GFile                     *file);
FOUNDRY_AVAILABLE_IN_ALL
void                      foundry_diagnostic_builder_set_path        (FoundryDiagnosticBuilder  *self,
                                                                      const char                *path);
FOUNDRY_AVAILABLE_IN_ALL
void                      foundry_diagnostic_builder_set_line        (FoundryDiagnosticBuilder  *self,
                                                                      guint                      line);
FOUNDRY_AVAILABLE_IN_ALL
void                      foundry_diagnostic_builder_set_line_offset (FoundryDiagnosticBuilder  *self,
                                                                      guint                      line_offset);
FOUNDRY_AVAILABLE_IN_ALL
void                      foundry_diagnostic_builder_add_range       (FoundryDiagnosticBuilder  *self,
                                                                      guint                      begin_line,
                                                                      guint                      begin_line_offset,
                                                                      guint                      end_line,
                                                                      guint                      end_line_offset);
FOUNDRY_AVAILABLE_IN_ALL
FoundryDiagnostic        *foundry_diagnostic_builder_end             (FoundryDiagnosticBuilder  *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_1_1
void                      foundry_diagnostic_builder_add_fix         (FoundryDiagnosticBuilder  *self,
                                                                      const char                *description,
                                                                      GListModel                *text_edits);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (FoundryDiagnosticBuilder, foundry_diagnostic_builder_unref)

G_END_DECLS
