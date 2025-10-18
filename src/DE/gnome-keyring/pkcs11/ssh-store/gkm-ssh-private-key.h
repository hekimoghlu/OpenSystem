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

#ifndef __GKM_SSH_PRIVATE_KEY_H__
#define __GKM_SSH_PRIVATE_KEY_H__

#include <glib-object.h>

#include "gkm-ssh-public-key.h"

#include "gkm/gkm-private-xsa-key.h"

#define GKM_TYPE_SSH_PRIVATE_KEY               (gkm_ssh_private_key_get_type ())
#define GKM_SSH_PRIVATE_KEY(obj)               (G_TYPE_CHECK_INSTANCE_CAST ((obj), GKM_TYPE_SSH_PRIVATE_KEY, GkmSshPrivateKey))
#define GKM_SSH_PRIVATE_KEY_CLASS(klass)       (G_TYPE_CHECK_CLASS_CAST ((klass), GKM_TYPE_SSH_PRIVATE_KEY, GkmSshPrivateKeyClass))
#define GKM_IS_SSH_PRIVATE_KEY(obj)            (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GKM_TYPE_SSH_PRIVATE_KEY))
#define GKM_IS_SSH_PRIVATE_KEY_CLASS(klass)    (G_TYPE_CHECK_CLASS_TYPE ((klass), GKM_TYPE_SSH_PRIVATE_KEY))
#define GKM_SSH_PRIVATE_KEY_GET_CLASS(obj)     (G_TYPE_INSTANCE_GET_CLASS ((obj), GKM_TYPE_SSH_PRIVATE_KEY, GkmSshPrivateKeyClass))

typedef struct _GkmSshPrivateKey GkmSshPrivateKey;
typedef struct _GkmSshPrivateKeyClass GkmSshPrivateKeyClass;

struct _GkmSshPrivateKeyClass {
	GkmPrivateXsaKeyClass parent_class;
};

GType               gkm_ssh_private_key_get_type               (void);

GkmSshPrivateKey*   gkm_ssh_private_key_new                    (GkmModule *module,
                                                                const gchar *unique);

gboolean            gkm_ssh_private_key_parse                  (GkmSshPrivateKey *self,
                                                                const gchar *public_path,
                                                                const gchar *private_path,
                                                                GError **error);

const gchar*        gkm_ssh_private_key_get_label              (GkmSshPrivateKey *key);

void                gkm_ssh_private_key_set_label              (GkmSshPrivateKey *key,
                                                                const gchar *label);

GkmSshPublicKey*    gkm_ssh_private_key_get_public_key         (GkmSshPrivateKey *self);

#endif /* __GKM_SSH_PRIVATE_KEY_H__ */
