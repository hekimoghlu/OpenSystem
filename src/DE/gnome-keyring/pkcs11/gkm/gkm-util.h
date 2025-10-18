/*
 * gnome-keyring
 *
 * Copyright (C) 2008 Stefan Walter
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef GKM_UTIL_H_
#define GKM_UTIL_H_

#include <glib.h>

#include <gcrypt.h>

#include "pkcs11/pkcs11.h"

typedef struct _GkmMemory {
	gconstpointer data;
	gsize n_data;
} GkmMemory;

GkmMemory*            gkm_util_memory_new                         (gconstpointer data,
                                                                   gsize n_data);

guint                 gkm_util_memory_hash                        (gconstpointer memory);

gboolean              gkm_util_memory_equal                       (gconstpointer memory_1,
                                                                   gconstpointer memory_2);

void                  gkm_util_memory_free                        (gpointer memory);

guint                 gkm_util_ulong_hash                         (gconstpointer ptr_to_ulong);

gboolean              gkm_util_ulong_equal                        (gconstpointer ptr_to_ulong_1,
                                                                   gconstpointer ptr_to_ulong_2);

gulong*               gkm_util_ulong_alloc                        (gulong value);

void                  gkm_util_ulong_free                         (gpointer ptr_to_ulong);

CK_RV                 gkm_util_return_data                        (CK_VOID_PTR output,
                                                                   CK_ULONG_PTR n_output,
                                                                   gconstpointer input,
                                                                   gsize n_input);

CK_RV                 gkm_attribute_set_mpi                       (CK_ATTRIBUTE_PTR attr,
                                                                   gcry_mpi_t mpi);

CK_ULONG              gkm_util_next_handle                        (void);

void                  gkm_util_dispose_unref                      (gpointer object);

gchar *               gkm_util_locate_keyrings_directory          (void);

#endif /* GKM_UTIL_H_ */
