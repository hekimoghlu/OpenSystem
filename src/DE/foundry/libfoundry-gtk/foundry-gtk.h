/* foundry-gtk.h
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
#include <gtksourceview/gtksource.h>

#define FOUNDRY_GTK_INSIDE
#include "foundry-changes-gutter-renderer.h"
#include "foundry-diagnostics-gutter-renderer.h"
#include "foundry-gtk-init.h"
#include "foundry-markup-view.h"
#include "foundry-menu-manager.h"
#include "foundry-menu-proxy.h"
#include "foundry-shortcut-bundle.h"
#include "foundry-shortcut-info.h"
#include "foundry-shortcut-manager.h"
#include "foundry-shortcut-observer.h"
#include "foundry-shortcut-provider.h"
#include "foundry-source-buffer.h"
#include "foundry-source-view.h"
#include "foundry-source-view-addin.h"
#include "foundry-terminal.h"
#include "foundry-terminal-palette.h"
#include "foundry-terminal-palette-set.h"
#undef FOUNDRY_GTK_INSIDE
