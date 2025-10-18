/* foundry-adw.h
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

#include <adwaita.h>
#include <foundry.h>
#include <foundry-gtk.h>

#define FOUNDRY_ADW_INSIDE
# include "foundry-adw-init.h"
# include "foundry-page.h"
# include "foundry-panel.h"
# include "foundry-panel-bar.h"
# include "foundry-search-dialog.h"
# include "foundry-tree-expander.h"
# include "foundry-workspace.h"
# include "foundry-workspace-addin.h"
#undef FOUNDRY_ADW_INSIDE
