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

#ifndef __GKM_DH_PUBLIC_KEY_H__
#define __GKM_DH_PUBLIC_KEY_H__

#include <glib-object.h>

#include "gkm-dh-key.h"
#include "gkm-types.h"

#define GKM_FACTORY_DH_PUBLIC_KEY            (gkm_dh_public_key_get_factory ())

#define GKM_TYPE_DH_PUBLIC_KEY               (gkm_dh_public_key_get_type ())
#define GKM_DH_PUBLIC_KEY(obj)               (G_TYPE_CHECK_INSTANCE_CAST ((obj), GKM_TYPE_DH_PUBLIC_KEY, GkmDhPublicKey))
#define GKM_DH_PUBLIC_KEY_CLASS(klass)       (G_TYPE_CHECK_CLASS_CAST ((klass), GKM_TYPE_DH_PUBLIC_KEY, GkmDhPublicKeyClass))
#define GKM_IS_DH_PUBLIC_KEY(obj)            (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GKM_TYPE_DH_PUBLIC_KEY))
#define GKM_IS_DH_PUBLIC_KEY_CLASS(klass)    (G_TYPE_CHECK_CLASS_TYPE ((klass), GKM_TYPE_DH_PUBLIC_KEY))
#define GKM_DH_PUBLIC_KEY_GET_CLASS(obj)     (G_TYPE_INSTANCE_GET_CLASS ((obj), GKM_TYPE_DH_PUBLIC_KEY, GkmDhPublicKeyClass))

typedef struct _GkmDhPublicKeyClass GkmDhPublicKeyClass;

struct _GkmDhPublicKeyClass {
	GkmDhKeyClass parent_class;
};

GType                     gkm_dh_public_key_get_type           (void);

GkmFactory*               gkm_dh_public_key_get_factory        (void);

GkmDhPublicKey*           gkm_dh_public_key_new                (GkmModule *module,
                                                                GkmManager *manager,
                                                                gcry_mpi_t prime,
                                                                gcry_mpi_t base,
                                                                gcry_mpi_t value,
                                                                gpointer id,
                                                                gsize n_id);

#endif /* __GKM_DH_PUBLIC_KEY_H__ */
