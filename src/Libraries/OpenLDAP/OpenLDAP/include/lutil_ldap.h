/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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
/* This work is part of OpenLDAP Software <http://www.openldap.org/>.
 *
 * Copyright 1998-2011 The OpenLDAP Foundation.
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

#ifndef _LUTIL_LDAP_H
#define _LUTIL_LDAP_H 1

#include <ldap_cdefs.h>
#include <lber_types.h>

/*
 * Include file for lutil LDAP routines
 */

LDAP_BEGIN_DECL

LDAP_LUTIL_F( void )
lutil_sasl_freedefs LDAP_P((
	void *defaults ));

LDAP_LUTIL_F( void * )
lutil_sasl_defaults LDAP_P((
	LDAP *ld,
	char *mech,
	char *realm,
	char *authcid,
	char *passwd,
	char *authzid ));

LDAP_LUTIL_F( int )
lutil_sasl_interact LDAP_P((
	LDAP *ld, unsigned flags, void *defaults, void *p ));

LDAP_END_DECL

#endif /* _LUTIL_LDAP_H */
