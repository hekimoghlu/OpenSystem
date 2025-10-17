/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 13, 2021.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/* $OpenLDAP$ */
/* This work is part of OpenLDAP Software <http://www.openldap.org/>.
 *
 * Copyright 2003-2011 The OpenLDAP Foundation.
 * Portions Copyright 2003-2010 Howard Chu.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted only as authorized by the OpenLDAP
 * Public License.
 *
 * A copy of this license is available in the file LICENSE in the
 * top-level directory of the distribution or, alternatively, at
 * <http://www.OpenLDAP.org/license.html>.
 */
/* ACKNOWLEDGEMENTS:
 * This work was initially developed by the Howard Chu for inclusion
 * in OpenLDAP Software.
 */

#include "portable.h"

#include <stdio.h>

#include <ac/string.h>
#include <ac/socket.h>

#include "lutil.h"
#include "slap.h"
#include "back-ldap.h"
#include "config.h"

static BackendInfo	*lback;

static slap_overinst ldappbind;

static int
ldap_pbind_bind(
	Operation	*op,
	SlapReply	*rs )
{
	slap_overinst	*on = (slap_overinst *) op->o_bd->bd_info;
	void *private = op->o_bd->be_private;
	void *bi = op->o_bd->bd_info;
	int rc;

	op->o_bd->bd_info = lback;
	op->o_bd->be_private = on->on_bi.bi_private;
	rc = lback->bi_op_bind( op, rs );
	op->o_bd->be_private = private;
	op->o_bd->bd_info = bi;

	return rc;
}

static int
ldap_pbind_db_init(
	BackendDB *be,
	ConfigReply *cr )
{
	slap_overinst	*on = (slap_overinst *)be->bd_info;
	ConfigOCs	*be_cf_ocs = be->be_cf_ocs;
	void		*private = be->be_private;
	int rc;

	if ( lback == NULL ) {
		lback = backend_info( "ldap" );

		if ( lback == NULL ) {
			return 1;
		}
	}

	rc = lback->bi_db_init( be, cr );
	on->on_bi.bi_private = be->be_private;
	be->be_cf_ocs = be_cf_ocs;
	be->be_private = private;

	return rc;
}

static int
ldap_pbind_db_open(
	BackendDB	*be,
	ConfigReply	*cr )
{
	slap_overinst	*on = (slap_overinst *) be->bd_info;
	void	*private = be->be_private;
	int		rc;
	int		monitoring;

    be->be_private = on->on_bi.bi_private;
	monitoring = ( SLAP_DBFLAGS( be ) & SLAP_DBFLAG_MONITORING );
	SLAP_DBFLAGS( be ) &= ~SLAP_DBFLAG_MONITORING;
	rc = lback->bi_db_open( be, cr );
	SLAP_DBFLAGS( be ) |= monitoring;
	be->be_private = private;

	return rc;
}

static int
ldap_pbind_db_close(
	BackendDB	*be,
	ConfigReply	*cr )
{
	slap_overinst	*on = (slap_overinst *) be->bd_info;
	void	*private = be->be_private;
	int		rc;

    be->be_private = on->on_bi.bi_private;
	rc = lback->bi_db_close( be, cr );
	be->be_private = private;

	return rc;
}

static int
ldap_pbind_db_destroy(
	BackendDB	*be,
	ConfigReply	*cr )
{
	slap_overinst	*on = (slap_overinst *) be->bd_info;
	void	*private = be->be_private;
	int		rc;

    be->be_private = on->on_bi.bi_private;
	rc = lback->bi_db_close( be, cr );
	on->on_bi.bi_private = be->be_private;
	be->be_private = private;

	return rc;
}

static int
ldap_pbind_connection_destroy(
	BackendDB *be,
	Connection *conn
)
{
	slap_overinst	*on = (slap_overinst *) be->bd_info;
	void			*private = be->be_private;
	int				rc;

	be->be_private = on->on_bi.bi_private;
	rc = lback->bi_connection_destroy( be, conn );
	be->be_private = private;

	return rc;
}

int
pbind_initialize( void )
{
	int rc;

	ldappbind.on_bi.bi_type = "pbind";
	ldappbind.on_bi.bi_db_init = ldap_pbind_db_init;
	ldappbind.on_bi.bi_db_open = ldap_pbind_db_open;
	ldappbind.on_bi.bi_db_close = ldap_pbind_db_close;
	ldappbind.on_bi.bi_db_destroy = ldap_pbind_db_destroy;

	ldappbind.on_bi.bi_op_bind = ldap_pbind_bind;
	ldappbind.on_bi.bi_connection_destroy = ldap_pbind_connection_destroy;

	rc = ldap_pbind_init_cf( &ldappbind.on_bi );
	if ( rc ) {
		return rc;
	}

	return overlay_register( &ldappbind );
}
