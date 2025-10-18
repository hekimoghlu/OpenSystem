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

#ifndef __GKM_SECRET_H__
#define __GKM_SECRET_H__

#include <glib-object.h>

#include "gkm-types.h"

#include "pkcs11/pkcs11.h"

#define GKM_TYPE_SECRET               (gkm_secret_get_type ())
#define GKM_SECRET(obj)               (G_TYPE_CHECK_INSTANCE_CAST ((obj), GKM_TYPE_SECRET, GkmSecret))
#define GKM_SECRET_CLASS(klass)       (G_TYPE_CHECK_CLASS_CAST ((klass), GKM_TYPE_SECRET, GkmSecretClass))
#define GKM_IS_SECRET(obj)            (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GKM_TYPE_SECRET))
#define GKM_IS_SECRET_CLASS(klass)    (G_TYPE_CHECK_CLASS_TYPE ((klass), GKM_TYPE_SECRET))
#define GKM_SECRET_GET_CLASS(obj)     (G_TYPE_INSTANCE_GET_CLASS ((obj), GKM_TYPE_SECRET, GkmSecretClass))

typedef struct _GkmSecretClass GkmSecretClass;

struct _GkmSecretClass {
	GObjectClass parent_class;
};

GType               gkm_secret_get_type               (void);

GkmSecret*          gkm_secret_new                    (const guchar *data,
                                                       gssize n_data);

GkmSecret*          gkm_secret_new_from_login         (CK_UTF8CHAR_PTR pin,
                                                       CK_ULONG n_pin);

GkmSecret*          gkm_secret_new_from_password      (const gchar *password);

const guchar*       gkm_secret_get                    (GkmSecret *self,
                                                       gsize *n_data);

const gchar*        gkm_secret_get_password           (GkmSecret *self,
                                                       gsize *n_pin);

gboolean            gkm_secret_equal                  (GkmSecret *self,
                                                       GkmSecret *other);

gboolean            gkm_secret_equals                 (GkmSecret *self,
                                                       const guchar *data,
                                                       gssize n_data);

gboolean            gkm_secret_is_trivially_weak      (GkmSecret *self);

#endif /* __GKM_SECRET_H__ */
