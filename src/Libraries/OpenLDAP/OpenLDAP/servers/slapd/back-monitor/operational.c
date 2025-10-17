/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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
 * Copyright 2001-2011 The OpenLDAP Foundation.
 * Portions Copyright 2001-2003 Pierangelo Masarati.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted only as authorized by the OpenLDAP
 * Public License.
 *
 * A copy of this license is available in file LICENSE in the
 * top-level directory of the distribution or, alternatively, at
 * <http://www.OpenLDAP.org/license.html>.
 */
/* ACKNOWLEDGEMENTS:
 * This work was initially developed by Pierangelo Masarati for inclusion
 * in OpenLDAP Software.
 */

#include "portable.h"

#include <stdio.h>

#include <ac/string.h>
#include <ac/socket.h>

#include "slap.h"
#include "back-monitor.h"
#include "proto-back-monitor.h"

/*
 * sets the supported operational attributes (if required)
 */

int
monitor_back_operational(
	Operation	*op,
	SlapReply	*rs )
{
	Attribute	**ap;

	assert( rs->sr_entry != NULL );

	for ( ap = &rs->sr_operational_attrs; *ap; ap = &(*ap)->a_next ) {
		if ( (*ap)->a_desc == slap_schema.si_ad_hasSubordinates ) {
			break;
		}
	}

	if ( *ap == NULL &&
		attr_find( rs->sr_entry->e_attrs, slap_schema.si_ad_hasSubordinates ) == NULL &&
		( SLAP_OPATTRS( rs->sr_attr_flags ) ||
			ad_inlist( slap_schema.si_ad_hasSubordinates, rs->sr_attrs ) ) )
	{
		int			hs;
		monitor_entry_t	*mp;

		mp = ( monitor_entry_t * )rs->sr_entry->e_private;

		assert( mp != NULL );

		hs = MONITOR_HAS_CHILDREN( mp );
		*ap = slap_operational_hasSubordinate( hs );
		assert( *ap != NULL );
		ap = &(*ap)->a_next;
	}
	
	return LDAP_SUCCESS;
}

