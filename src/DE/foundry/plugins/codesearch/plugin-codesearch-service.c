/* plugin-codesearch-service.c
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

#include "config.h"

#include "plugin-codesearch-service.h"

struct _PluginCodesearchService
{
  FoundryService parent_instance;
};

struct _PluginCodesearchServiceClass
{
  FoundryServiceClass parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginCodesearchService, plugin_codesearch_service, FOUNDRY_TYPE_SERVICE)

static DexFuture *
plugin_codesearch_service_start (FoundryService *service)
{
  g_assert (PLUGIN_IS_CODESEARCH_SERVICE (service));

  return dex_future_new_true ();
}

static DexFuture *
plugin_codesearch_service_stop (FoundryService *service)
{
  g_assert (PLUGIN_IS_CODESEARCH_SERVICE (service));

  return dex_future_new_true ();
}

static void
plugin_codesearch_service_class_init (PluginCodesearchServiceClass *klass)
{
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  service_class->start = plugin_codesearch_service_start;
  service_class->stop = plugin_codesearch_service_stop;
}

static void
plugin_codesearch_service_init (PluginCodesearchService *self)
{
}
