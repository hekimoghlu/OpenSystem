/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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
 * Copyright 1998-2011 The OpenLDAP Foundation.
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
/* Portions Copyright (c) 1990 Regents of the University of Michigan.
 * All rights reserved.
 */

#include "portable.h"

#include <stdio.h>

#include <ac/stdlib.h>

#include <ac/socket.h>
#include <ac/string.h>
#include <ac/time.h>

#include "ldap-int.h"

LDAPMessage *
ldap_delete_result_entry( LDAPMessage **list, LDAPMessage *e )
{
	LDAPMessage	*tmp, *prev = NULL;

	assert( list != NULL );
	assert( e != NULL );

	for ( tmp = *list; tmp != NULL && tmp != e; tmp = tmp->lm_chain )
		prev = tmp;

	if ( tmp == NULL )
		return( NULL );

	if ( prev == NULL ) {
		if ( tmp->lm_chain )
			tmp->lm_chain->lm_chain_tail = (*list)->lm_chain_tail;
		*list = tmp->lm_chain;
	} else {
		prev->lm_chain = tmp->lm_chain;
		if ( prev->lm_chain == NULL )
			(*list)->lm_chain_tail = prev;
	}
	tmp->lm_chain = NULL;

	return( tmp );
}

void
ldap_add_result_entry( LDAPMessage **list, LDAPMessage *e )
{
	assert( list != NULL );
	assert( e != NULL );

	e->lm_chain = *list;
	if ( *list )
		e->lm_chain_tail = (*list)->lm_chain_tail;
	else
		e->lm_chain_tail = e;
	*list = e;
}
