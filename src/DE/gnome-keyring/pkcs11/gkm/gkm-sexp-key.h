/*
 * gnome-sexp_keyring
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

#ifndef __GKM_SEXP_KEY_H__
#define __GKM_SEXP_KEY_H__

#include <glib-object.h>

#include "gkm-sexp.h"
#include "gkm-object.h"
#include "gkm-types.h"

#define GKM_TYPE_SEXP_KEY               (gkm_sexp_key_get_type ())
#define GKM_SEXP_KEY(obj)               (G_TYPE_CHECK_INSTANCE_CAST ((obj), GKM_TYPE_SEXP_KEY, GkmSexpKey))
#define GKM_SEXP_KEY_CLASS(klass)       (G_TYPE_CHECK_CLASS_CAST ((klass), GKM_TYPE_SEXP_KEY, GkmSexpKeyClass))
#define GKM_IS_SEXP_KEY(obj)            (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GKM_TYPE_SEXP_KEY))
#define GKM_IS_SEXP_KEY_CLASS(klass)    (G_TYPE_CHECK_CLASS_TYPE ((klass), GKM_TYPE_SEXP_KEY))
#define GKM_SEXP_KEY_GET_CLASS(obj)     (G_TYPE_INSTANCE_GET_CLASS ((obj), GKM_TYPE_SEXP_KEY, GkmSexpKeyClass))

typedef struct _GkmSexpKeyClass GkmSexpKeyClass;
typedef struct _GkmSexpKeyPrivate GkmSexpKeyPrivate;

struct _GkmSexpKey {
	GkmObject parent;
	GkmSexpKeyPrivate *pv;
};

struct _GkmSexpKeyClass {
	GkmObjectClass parent_class;

	/* virtual methods */

	GkmSexp* (*acquire_crypto_sexp) (GkmSexpKey *self, GkmSession *session);
};

GType                gkm_sexp_key_get_type                (void);

GkmSexp*             gkm_sexp_key_get_base                (GkmSexpKey *self);

void                 gkm_sexp_key_set_base                (GkmSexpKey *self,
                                                           GkmSexp *sexp);

int                  gkm_sexp_key_get_algorithm           (GkmSexpKey *self);

CK_RV                gkm_sexp_key_set_part                (GkmSexpKey *self,
                                                           int algorithm,
                                                           const char *part,
                                                           CK_ATTRIBUTE_PTR attr);

CK_RV                gkm_sexp_key_set_ec_params           (GkmSexpKey *self,
                                                           int algo,
                                                           CK_ATTRIBUTE_PTR attr);

CK_RV                gkm_sexp_key_set_ec_q                (GkmSexpKey *self,
                                                           int algo,
                                                           CK_ATTRIBUTE_PTR attr);

GkmSexp*             gkm_sexp_key_acquire_crypto_sexp     (GkmSexpKey *self,
                                                           GkmSession *session);

#endif /* __GKM_SEXP_KEY_H__ */
