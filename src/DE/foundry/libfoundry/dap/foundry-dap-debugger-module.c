/* foundry-dap-debugger-module.c
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

#include "foundry-dap-debugger-module-private.h"
#include "foundry-debugger-mapped-region.h"

struct _FoundryDapDebuggerModule
{
  FoundryDebuggerModule parent_instance;
  GWeakRef debugger_wr;
  char *id;
  char *name;
  char *path;
};

G_DEFINE_FINAL_TYPE (FoundryDapDebuggerModule, foundry_dap_debugger_module, FOUNDRY_TYPE_DEBUGGER_MODULE)

static char *
foundry_dap_debugger_module_dup_id (FoundryDebuggerModule *module)
{
  FoundryDapDebuggerModule *self = FOUNDRY_DAP_DEBUGGER_MODULE (module);

  return g_strdup (self->id);
}

static char *
foundry_dap_debugger_module_dup_name (FoundryDebuggerModule *module)
{
  FoundryDapDebuggerModule *self = FOUNDRY_DAP_DEBUGGER_MODULE (module);

  return g_strdup (self->name);
}

static char *
foundry_dap_debugger_module_dup_path (FoundryDebuggerModule *module)
{
  FoundryDapDebuggerModule *self = FOUNDRY_DAP_DEBUGGER_MODULE (module);

  return g_strdup (self->path);
}

static GListModel *
foundry_dap_debugger_module_list_address_space (FoundryDebuggerModule *module)
{
  FoundryDapDebuggerModule *self = FOUNDRY_DAP_DEBUGGER_MODULE (module);
  g_autoptr(FoundryDebugger) debugger = NULL;
  g_autoptr(GListModel) address_space = NULL;
  g_autoptr(GListStore) store = g_list_store_new (FOUNDRY_TYPE_DEBUGGER_MAPPED_REGION);

  if (self->path != NULL &&
      (debugger = g_weak_ref_get (&self->debugger_wr)) &&
      (address_space = foundry_debugger_list_address_space (debugger)))
    {
      guint n_items = g_list_model_get_n_items (address_space);

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryDebuggerMappedRegion) region = g_list_model_get_item (address_space, i);
          g_autofree char *path = foundry_debugger_mapped_region_dup_path (region);

          if (g_strcmp0 (path, self->path) == 0)
            g_list_store_append (store, region);
        }
    }

  return G_LIST_MODEL (g_steal_pointer (&store));
}

static void
foundry_dap_debugger_module_finalize (GObject *object)
{
  FoundryDapDebuggerModule *self = (FoundryDapDebuggerModule *)object;

  g_clear_pointer (&self->id, g_free);
  g_clear_pointer (&self->name, g_free);
  g_clear_pointer (&self->path, g_free);
  g_weak_ref_clear (&self->debugger_wr);

  G_OBJECT_CLASS (foundry_dap_debugger_module_parent_class)->finalize (object);
}

static void
foundry_dap_debugger_module_class_init (FoundryDapDebuggerModuleClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDebuggerModuleClass *module_class = FOUNDRY_DEBUGGER_MODULE_CLASS (klass);

  object_class->finalize = foundry_dap_debugger_module_finalize;

  module_class->dup_id = foundry_dap_debugger_module_dup_id;
  module_class->dup_name = foundry_dap_debugger_module_dup_name;
  module_class->dup_path = foundry_dap_debugger_module_dup_path;
  module_class->list_address_space = foundry_dap_debugger_module_list_address_space;
}

static void
foundry_dap_debugger_module_init (FoundryDapDebuggerModule *self)
{
  g_weak_ref_init (&self->debugger_wr, NULL);
}

FoundryDebuggerModule *
foundry_dap_debugger_module_new (FoundryDapDebugger *debugger,
                                 const char         *id,
                                 const char         *name,
                                 const char         *path)
{
  FoundryDapDebuggerModule *self;

  g_return_val_if_fail (FOUNDRY_IS_DAP_DEBUGGER (debugger), NULL);

  self = g_object_new (FOUNDRY_TYPE_DAP_DEBUGGER_MODULE, NULL);
  g_weak_ref_set (&self->debugger_wr, debugger);
  self->id = g_strdup (id);
  self->name = g_strdup (name);
  self->path = g_strdup (path);

  return FOUNDRY_DEBUGGER_MODULE (self);
}
