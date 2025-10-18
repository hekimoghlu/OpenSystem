/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 8; tab-width: 8 -*- */
/*
 * GData Client
 * Copyright (C) Thibault Saunier 2009 <saunierthibault@gmail.com>
 * Copyright (C) Philip Withnall 2009 <philip@tecnocode.co.uk>
 *
 * GData Client is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * GData Client is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GData Client.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef GDATA_ACCESS_RULE_H
#define GDATA_ACCESS_RULE_H

#include <glib.h>
#include <glib-object.h>

#include <gdata/gdata-entry.h>

G_BEGIN_DECLS

/**
 * GDATA_ACCESS_SCOPE_USER:
 *
 * The #GDataAccessRule applies to a single individual, whose e-mail address is given in #GDataAccessRule:scope-value.
 *
 * Since: 0.7.0
 */
#define GDATA_ACCESS_SCOPE_USER "user"

/**
 * GDATA_ACCESS_SCOPE_DOMAIN:
 *
 * The #GDataAccessRule applies to all users in a Google Apps For Your Domain domain, given in #GDataAccessRule:scope-value.
 *
 * Since: 0.7.0
 */
#define GDATA_ACCESS_SCOPE_DOMAIN "domain"

/**
 * GDATA_ACCESS_SCOPE_DEFAULT:
 *
 * The #GDataAccessRule applies to all users.
 *
 * Since: 0.7.0
 */
#define GDATA_ACCESS_SCOPE_DEFAULT "default"

/**
 * GDATA_ACCESS_ROLE_NONE:
 *
 * The users specified by the #GDataAccessRule have no rights.
 *
 * Since: 0.7.0
 */
#define GDATA_ACCESS_ROLE_NONE "none"

#define GDATA_TYPE_ACCESS_RULE		(gdata_access_rule_get_type ())
#define GDATA_ACCESS_RULE(o)		(G_TYPE_CHECK_INSTANCE_CAST ((o), GDATA_TYPE_ACCESS_RULE, GDataAccessRule))
#define GDATA_ACCESS_RULE_CLASS(k)	(G_TYPE_CHECK_CLASS_CAST((k), GDATA_TYPE_ACCESS_RULE, GDataAccessRuleClass))
#define GDATA_IS_ACCESS_RULE(o)		(G_TYPE_CHECK_INSTANCE_TYPE ((o), GDATA_TYPE_ACCESS_RULE))
#define GDATA_IS_ACCESS_RULE_CLASS(k)	(G_TYPE_CHECK_CLASS_TYPE ((k), GDATA_TYPE_ACCESS_RULE))
#define GDATA_ACCESS_RULE_GET_CLASS(o)	(G_TYPE_INSTANCE_GET_CLASS ((o), GDATA_TYPE_ACCESS_RULE, GDataAccessRuleClass))

typedef struct _GDataAccessRulePrivate	GDataAccessRulePrivate;

/**
 * GDataAccessRule:
 *
 * All the fields in the #GDataAccessRule structure are private and should never be accessed directly.
 *
 * Since: 0.3.0
 */
typedef struct {
	GDataEntry parent;
	GDataAccessRulePrivate *priv;
} GDataAccessRule;

/**
 * GDataAccessRuleClass:
 *
 * All the fields in the #GDataAccessRuleClass structure are private and should never be accessed directly.
 *
 * Since: 0.3.0
 */
typedef struct {
	/*< private >*/
	GDataEntryClass parent;

	/*< private >*/
	/* Padding for future expansion */
	void (*_g_reserved0) (void);
	void (*_g_reserved1) (void);
} GDataAccessRuleClass;

GType gdata_access_rule_get_type (void) G_GNUC_CONST;

GDataAccessRule *gdata_access_rule_new (const gchar *id) G_GNUC_WARN_UNUSED_RESULT G_GNUC_MALLOC;

const gchar *gdata_access_rule_get_role (GDataAccessRule *self) G_GNUC_PURE;
void gdata_access_rule_set_role (GDataAccessRule *self, const gchar *role);
void gdata_access_rule_get_scope (GDataAccessRule *self, const gchar **type, const gchar **value);
void gdata_access_rule_set_scope (GDataAccessRule *self, const gchar *type, const gchar *value);
gint64 gdata_access_rule_get_edited (GDataAccessRule *self);
const gchar *gdata_access_rule_get_key (GDataAccessRule *self) G_GNUC_PURE;

G_END_DECLS

#endif /* !GDATA_ACCESS_RULE_H */
