/* plugin-qemu-device-provider.c
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

#include <glib/gi18n-lib.h>

#include "line-reader-private.h"

#include "plugin-qemu-device-provider.h"

struct _PluginQemuDeviceProvider
{
  FoundryDeviceProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginQemuDeviceProvider, plugin_qemu_device_provider, FOUNDRY_TYPE_DEVICE_PROVIDER)

static const struct {
  const char *filename;
  const char *arch;
  const char *suffix;
} machines[] = {
  /* translators: format is "CPU emulation". Only translate "emulation" */
  { "qemu-aarch64", "aarch64", N_("Aarch64 Emulation") },
  { "qemu-arm",     "arm",     N_("Arm Emulation") },
  { "qemu-riscv64", "riscv64", N_("riscv64 Emulation") },
  { "qemu-x86_64",  "x86_64",  N_("x64_64 Emulation") },
};

static gboolean
has_flag (GBytes *bytes,
          char    flag)
{
  const char *contents = g_bytes_get_data (bytes, NULL);
  gsize len = g_bytes_get_size (bytes);
  LineReader reader;
  const char *line;
  gsize line_len = 0;

  line_reader_init (&reader, (gchar *)contents, len);

  while ((line = line_reader_next (&reader, &line_len)))
    {
      if (strncmp (line, "flags: ", 7) == 0)
        {
          for (gsize i = 7; i < line_len; i++)
            {
              if (line[i] == flag)
                return TRUE;
            }
        }
    }

  return FALSE;
}

static gboolean
is_enabled (GBytes *bytes)
{
  const char *contents = g_bytes_get_data (bytes, NULL);
  gsize len = g_bytes_get_size (bytes);

  return memmem (contents, len, "enabled", 7) != NULL;
}

static DexFuture *
plugin_qemu_device_provider_load_fiber (gpointer user_data)
{
  PluginQemuDeviceProvider *self = user_data;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GBytes) mounts_bytes = NULL;
  g_autoptr(GBytes) status_bytes = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (PLUGIN_IS_QEMU_DEVICE_PROVIDER (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  if (!(mounts_bytes = dex_await_boxed (foundry_host_file_get_contents_bytes ("/proc/mounts"), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (NULL == memmem (g_bytes_get_data (mounts_bytes, NULL),
                      g_bytes_get_size (mounts_bytes),
                      "binfmt", 6))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_SUPPORTED,
                                  "Host system does not support binfmt");

  if (!(status_bytes = dex_await_boxed (foundry_host_file_get_contents_bytes ("/proc/sys/fs/binfmt_misc/status"), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!is_enabled (status_bytes))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_SUPPORTED,
                                  "binfmt hooks are not currently enabled");

  for (guint i = 0; i < G_N_ELEMENTS (machines); i++)
    {
      g_autofree char *path = g_build_filename ("/proc/sys/fs/binfmt_misc", machines[i].filename, NULL);
      g_autoptr(FoundryTriplet) triplet = NULL;
      g_autoptr(FoundryDevice) device = NULL;
      g_autoptr(GBytes) bytes = NULL;
      g_autofree char *id = NULL;
      g_autofree char *title = NULL;

      if (!(bytes = dex_await_boxed (foundry_host_file_get_contents_bytes (path), NULL)))
        continue;

      if (!is_enabled (bytes))
        continue;

      if (!has_flag (bytes, 'F'))
        continue;

      g_debug ("Discovered Qemu device \"%s\"", machines[i].arch);

      /* translators: first %s is replaced with hostname, second %s with the CPU architecture */
      title = g_strdup_printf (_("My Computer (%s)"),
                               machines[i].suffix);
      triplet = foundry_triplet_new (machines[i].arch);
      id = g_strdup_printf ("qemu:%s", machines[i].arch);

      device = foundry_local_device_new_full (context, id, title, triplet);

      foundry_device_provider_device_added (FOUNDRY_DEVICE_PROVIDER (self), device);
    }

  return dex_future_new_true ();
}

static DexFuture *
plugin_qemu_device_provider_load (FoundryDeviceProvider *provider)
{
  g_assert (PLUGIN_IS_QEMU_DEVICE_PROVIDER (provider));

  return dex_scheduler_spawn (NULL, 0,
                              plugin_qemu_device_provider_load_fiber,
                              g_object_ref (provider),
                              g_object_unref);
}

static DexFuture *
plugin_qemu_device_provider_unload (FoundryDeviceProvider *provider)
{
  g_assert (PLUGIN_IS_QEMU_DEVICE_PROVIDER (provider));

  return dex_future_new_true ();
}

static void
plugin_qemu_device_provider_class_init (PluginQemuDeviceProviderClass *klass)
{
  FoundryDeviceProviderClass *device_provider_class = FOUNDRY_DEVICE_PROVIDER_CLASS (klass);

  device_provider_class->load = plugin_qemu_device_provider_load;
  device_provider_class->unload = plugin_qemu_device_provider_unload;
}

static void
plugin_qemu_device_provider_init (PluginQemuDeviceProvider *self)
{
}
