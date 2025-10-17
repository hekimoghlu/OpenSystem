/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
 * Copyright 2007-2011 The OpenLDAP Foundation.
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
 * This work was initially developed by Brian Candler for inclusion
 * in OpenLDAP Software.
 */

#include "portable.h"

#include <stdio.h>

#include <ac/socket.h>

#include "slap.h"
#include "back-sock.h"

int
sock_back_initialize(
    BackendInfo	*bi
)
{
	bi->bi_open = 0;
	bi->bi_config = 0;
	bi->bi_close = 0;
	bi->bi_destroy = 0;

	bi->bi_db_init = sock_back_db_init;
	bi->bi_db_config = 0;
	bi->bi_db_open = 0;
	bi->bi_db_close = 0;
	bi->bi_db_destroy = sock_back_db_destroy;

	bi->bi_op_bind = sock_back_bind;
	bi->bi_op_unbind = sock_back_unbind;
	bi->bi_op_search = sock_back_search;
	bi->bi_op_compare = sock_back_compare;
	bi->bi_op_modify = sock_back_modify;
	bi->bi_op_modrdn = sock_back_modrdn;
	bi->bi_op_add = sock_back_add;
	bi->bi_op_delete = sock_back_delete;
	bi->bi_op_abandon = 0;

	bi->bi_extended = 0;

	bi->bi_chk_referrals = 0;

	bi->bi_connection_init = 0;
	bi->bi_connection_destroy = 0;

	return sock_back_init_cf( bi );
}

int
sock_back_db_init(
    Backend	*be,
	struct config_reply_s *cr
)
{
	struct sockinfo	*si;

	si = (struct sockinfo *) ch_calloc( 1, sizeof(struct sockinfo) );

	be->be_private = si;
	be->be_cf_ocs = be->bd_info->bi_cf_ocs;

	return si == NULL;
}

int
sock_back_db_destroy(
    Backend	*be,
	struct config_reply_s *cr
)
{
	free( be->be_private );
	return 0;
}

#if SLAPD_SOCK == SLAPD_MOD_DYNAMIC

/* conditionally define the init_module() function */
SLAP_BACKEND_INIT_MODULE( sock )

#endif /* SLAPD_SOCK == SLAPD_MOD_DYNAMIC */
