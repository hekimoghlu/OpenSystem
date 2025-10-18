/* foundry-plugin.h
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

#include <libpeas.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_PLUGIN_DEFINE(func, ...)                \
  _FOUNDRY_EXTERN void func (PeasObjectModule *module); \
  _FOUNDRY_EXTERN void func (PeasObjectModule *module)  \
  { __VA_ARGS__ }

#define FOUNDRY_PLUGIN_REGISTER_TYPE(type, plugin_type) \
  peas_object_module_register_extension_type (module, type, plugin_type);

G_END_DECLS
