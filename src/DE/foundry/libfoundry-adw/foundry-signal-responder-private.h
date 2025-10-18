/* foundry-signal-responder.h
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

#include <gtk/gtk.h>

#include "foundry-responder-private.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_SIGNAL_RESPONDER (foundry_signal_responder_get_type())

G_DECLARE_FINAL_TYPE (FoundrySignalResponder, foundry_signal_responder, FOUNDRY, SIGNAL_RESPONDER, FoundryResponder)

FoundrySignalResponder *foundry_signal_responder_new (GObject         *object,
                                                      const char      *detailed_signal,
                                                      gboolean         after,
                                                      GtkExpression   *return_value,
                                                      FoundryReaction *reaction);

G_END_DECLS
