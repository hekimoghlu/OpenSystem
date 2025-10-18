/* foundry-triplet.c
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

#include "config.h"

#include "foundry-triplet-private.h"
#include "foundry-util-private.h"

G_DEFINE_BOXED_TYPE (FoundryTriplet, foundry_triplet, foundry_triplet_ref, foundry_triplet_unref)

struct _FoundryTriplet
{
  volatile gint ref_count;

  char *full_name;
  char *arch;
  char *vendor;
  char *kernel;
  char *operating_system;
};

static FoundryTriplet *
_foundry_triplet_construct (void)
{
  FoundryTriplet *self;

  self = g_slice_new0 (FoundryTriplet);
  self->ref_count = 1;
  self->full_name = NULL;
  self->arch = NULL;
  self->vendor = NULL;
  self->kernel = NULL;
  self->operating_system = NULL;

  return self;
}

/**
 * foundry_triplet_new:
 * @full_name: The complete identifier of the machine
 *
 * Creates a new #FoundryTriplet from a given identifier. This identifier
 * can be a simple architecture name, a duet of "arch-kernel" (like "m68k-coff"), a triplet
 * of "arch-kernel-os" (like "x86_64-linux-gnu") or a quadriplet of "arch-vendor-kernel-os"
 * (like "i686-pc-linux-gnu")
 *
 * Returns: (transfer full): An #FoundryTriplet.
 */
FoundryTriplet *
foundry_triplet_new (const char *full_name)
{
  FoundryTriplet *self;
  g_auto (GStrv) parts = NULL;
  guint parts_length = 0;

  g_return_val_if_fail (full_name != NULL, NULL);

  self = _foundry_triplet_construct ();
  self->full_name = g_strdup (full_name);

  parts = g_strsplit (full_name, "-", 4);
  parts_length = g_strv_length (parts);
  /* Currently they can't have more than 4 parts */
  if (parts_length >= 4)
    {
      self->arch = g_strdup (parts[0]);
      self->vendor = g_strdup (parts[1]);
      self->kernel = g_strdup (parts[2]);
      self->operating_system = g_strdup (parts[3]);
    }
  else if (parts_length == 3)
    {
      self->arch = g_strdup (parts[0]);
      self->kernel = g_strdup (parts[1]);
      self->operating_system = g_strdup (parts[2]);
    }
  else if (parts_length == 2)
    {
      self->arch = g_strdup (parts[0]);
      self->kernel = g_strdup (parts[1]);
    }
  else if (parts_length == 1)
    self->arch = g_strdup (parts[0]);

  return self;
}

/**
 * foundry_triplet_new_from_system:
 *
 * Creates a new #FoundryTriplet from a the current system information
 *
 * Returns: (transfer full): An #FoundryTriplet.
 */
FoundryTriplet *
foundry_triplet_new_from_system (void)
{
  static FoundryTriplet *system_triplet;

  if (g_once_init_enter (&system_triplet))
    g_once_init_leave (&system_triplet, foundry_triplet_new (_foundry_get_system_type ()));

  return foundry_triplet_ref (system_triplet);
}

/**
 * foundry_triplet_new_with_triplet:
 * @arch: The name of the architecture of the machine (like "x86_64")
 * @kernel: (nullable): The name of the kernel of the machine (like "linux")
 * @operating_system: (nullable): The name of the os of the machine
 * (like "gnuabi64")
 *
 * Creates a new #FoundryTriplet from a given triplet of "arch-kernel-os"
 * (like "x86_64-linux-gnu")
 *
 * Returns: (transfer full): An #FoundryTriplet.
 */
FoundryTriplet *
foundry_triplet_new_with_triplet (const char *arch,
                              const char *kernel,
                              const char *operating_system)
{
  FoundryTriplet *self;
  g_autofree char *full_name = NULL;

  g_return_val_if_fail (arch != NULL, NULL);

  self = _foundry_triplet_construct ();
  self->arch = g_strdup (arch);
  self->kernel = g_strdup (kernel);
  self->operating_system = g_strdup (operating_system);

  full_name = g_strdup (arch);
  if (kernel != NULL)
    {
      g_autofree char *start_full_name = full_name;
      full_name = g_strdup_printf ("%s-%s", start_full_name, kernel);
    }

  if (operating_system != NULL)
    {
      g_autofree char *start_full_name = full_name;
      full_name = g_strdup_printf ("%s-%s", start_full_name, operating_system);
    }

  self->full_name = g_steal_pointer (&full_name);

  return self;
}

/**
 * foundry_triplet_new_with_quadruplet:
 * @arch: The name of the architecture of the machine (like "x86_64")
 * @vendor: (nullable): The name of the vendor of the machine (like "pc")
 * @kernel: (nullable): The name of the kernel of the machine (like "linux")
 * @operating_system: (nullable): The name of the os of the machine (like "gnuabi64")
 *
 * Creates a new #FoundryTriplet from a given quadruplet of
 * "arch-vendor-kernel-os" (like "i686-pc-linux-gnu")
 *
 * Returns: (transfer full): An #FoundryTriplet.
 */
FoundryTriplet *
foundry_triplet_new_with_quadruplet (const char *arch,
                                 const char *vendor,
                                 const char *kernel,
                                 const char *operating_system)
{
  FoundryTriplet *self;
  g_autofree char *full_name = NULL;

  g_return_val_if_fail (arch != NULL, NULL);

  if (vendor == NULL)
    return foundry_triplet_new_with_triplet (arch, kernel, operating_system);

  self = _foundry_triplet_construct ();
  self->arch = g_strdup (arch);
  self->vendor = g_strdup (vendor);
  self->kernel = g_strdup (kernel);
  self->operating_system = g_strdup (operating_system);

  full_name = g_strdup_printf ("%s-%s", arch, vendor);
  if (kernel != NULL)
    {
      g_autofree char *start_full_name = full_name;
      full_name = g_strdup_printf ("%s-%s", start_full_name, kernel);
    }

  if (operating_system != NULL)
    {
      g_autofree char *start_full_name = full_name;
      full_name = g_strdup_printf ("%s-%s", start_full_name, operating_system);
    }

  self->full_name = g_steal_pointer (&full_name);

  return self;
}

static void
foundry_triplet_finalize (FoundryTriplet *self)
{
  g_free (self->full_name);
  g_free (self->arch);
  g_free (self->vendor);
  g_free (self->kernel);
  g_free (self->operating_system);
  g_slice_free (FoundryTriplet, self);
}

/**
 * foundry_triplet_ref:
 * @self: An #FoundryTriplet
 *
 * Increases the reference count of @self
 *
 * Returns: (transfer none): An #FoundryTriplet.
 */
FoundryTriplet *
foundry_triplet_ref (FoundryTriplet *self)
{
  g_return_val_if_fail (self, NULL);
  g_return_val_if_fail (self->ref_count > 0, NULL);

  g_atomic_int_inc (&self->ref_count);

  return self;
}

/**
 * foundry_triplet_unref:
 * @self: An #FoundryTriplet
 *
 * Decreases the reference count of @self
 * Once the reference count reaches 0, the object is freed.
 */
void
foundry_triplet_unref (FoundryTriplet *self)
{
  g_return_if_fail (self);
  g_return_if_fail (self->ref_count > 0);

  if (g_atomic_int_dec_and_test (&self->ref_count))
    foundry_triplet_finalize (self);
}

/**
 * foundry_triplet_get_full_name:
 * @self: An #FoundryTriplet
 *
 * Gets the full name of the machine configuration name (can be an architecture name,
 * a duet, a triplet or a quadruplet).
 *
 * Returns: (transfer none): The full name of the machine configuration name
 */
const char *
foundry_triplet_get_full_name (FoundryTriplet *self)
{
  g_return_val_if_fail (self, NULL);

  return self->full_name;
}

/**
 * foundry_triplet_get_arch:
 * @self: An #FoundryTriplet
 *
 * Gets the architecture name of the machine
 *
 * Returns: (transfer none): The architecture name of the machine
 */
const char *
foundry_triplet_get_arch (FoundryTriplet *self)
{
  g_return_val_if_fail (self, NULL);

  return self->arch;
}

/**
 * foundry_triplet_get_vendor:
 * @self: An #FoundryTriplet
 *
 * Gets the vendor name of the machine
 *
 * Returns: (transfer none) (nullable): The vendor name of the machine
 */
const char *
foundry_triplet_get_vendor (FoundryTriplet *self)
{
  g_return_val_if_fail (self, NULL);

  return self->vendor;
}

/**
 * foundry_triplet_get_kernel:
 * @self: An #FoundryTriplet
 *
 * Gets name of the kernel of the machine
 *
 * Returns: (transfer none) (nullable): The name of the kernel of the machine
 */
const char *
foundry_triplet_get_kernel (FoundryTriplet *self)
{
  g_return_val_if_fail (self, NULL);

  return self->kernel;
}

/**
 * foundry_triplet_get_operating_system:
 * @self: An #FoundryTriplet
 *
 * Gets name of the operating system of the machine
 *
 * Returns: (transfer none) (nullable): The name of the operating system of the machine
 */
const char *
foundry_triplet_get_operating_system (FoundryTriplet *self)
{
  g_return_val_if_fail (self, NULL);

  return self->operating_system;
}

/**
 * foundry_triplet_is_system:
 * @self: An #FoundryTriplet
 *
 * Gets whether this is the same architecture as the system
 *
 * Returns: %TRUE if this is the same architecture as the system, %FALSE otherwise
 */
gboolean
foundry_triplet_is_system (FoundryTriplet *self)
{
  g_autofree char *system_arch = _foundry_get_system_arch ();

  g_return_val_if_fail (self, FALSE);

  return g_strcmp0 (self->arch, system_arch) == 0;
}

JsonNode *
_foundry_triplet_to_json (gconstpointer data)
{
  const FoundryTriplet *self = data;
  JsonNode *node = json_node_new (JSON_NODE_VALUE);
  json_node_set_string (node, self->full_name);
  return node;
}
