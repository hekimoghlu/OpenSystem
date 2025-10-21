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

#ifndef __GKM_DH_KEY_H__
#define __GKM_DH_KEY_H__

#include <glib-object.h>

#include "gkm-object.h"
#include "gkm-types.h"

#define GKM_TYPE_DH_KEY               (gkm_dh_key_get_type ())
#define GKM_DH_KEY(obj)               (G_TYPE_CHECK_INSTANCE_CAST ((obj), GKM_TYPE_DH_KEY, GkmDhKey))
#define GKM_DH_KEY_CLASS(klass)       (G_TYPE_CHECK_CLASS_CAST ((klass), GKM_TYPE_DH_KEY, GkmDhKeyClass))
#define GKM_IS_DH_KEY(obj)            (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GKM_TYPE_DH_KEY))
#define GKM_IS_DH_KEY_CLASS(klass)    (G_TYPE_CHECK_CLASS_TYPE ((klass), GKM_TYPE_DH_KEY))
#define GKM_DH_KEY_GET_CLASS(obj)     (G_TYPE_INSTANCE_GET_CLASS ((obj), GKM_TYPE_DH_KEY, GkmDhKeyClass))

typedef struct _GkmDhKeyClass GkmDhKeyClass;
typedef struct _GkmDhKeyPrivate GkmDhKeyPrivate;

struct _GkmDhKey {
	GkmObject parent;
	GkmDhKeyPrivate *pv;
};

struct _GkmDhKeyClass {
	GkmObjectClass parent_class;
};

GType                     gkm_dh_key_get_type           (void);

void                      gkm_dh_key_initialize         (GkmDhKey *self,
                                                         gcry_mpi_t prime,
                                                         gcry_mpi_t base,
                                                         gpointer id,
                                                         gsize n_id);

gcry_mpi_t                gkm_dh_key_get_prime          (GkmDhKey *self);

#endif /* __GKM_DH_KEY_H__ */
