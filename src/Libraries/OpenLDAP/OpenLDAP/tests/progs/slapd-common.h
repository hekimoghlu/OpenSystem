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
 * Copyright 1999-2011 The OpenLDAP Foundation.
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
 * This work was initially developed by Howard Chu for inclusion
 * in OpenLDAP Software.
 */

#ifndef SLAPD_COMMON_H
#define SLAPD_COMMON_H

typedef enum {
	TESTER_TESTER,
	TESTER_ADDEL,
	TESTER_BIND,
	TESTER_MODIFY,
	TESTER_MODRDN,
	TESTER_READ,
	TESTER_SEARCH,
	TESTER_LAST
} tester_t;

extern void tester_init( const char *pname, tester_t ptype );
extern char * tester_uri( char *uri, char *host, int port );
extern void tester_error( const char *msg );
extern void tester_perror( const char *fname, const char *msg );
extern void tester_ldap_error( LDAP *ld, const char *fname, const char *msg );
extern int tester_ignore_str2errlist( const char *err );
extern int tester_ignore_err( int err );

extern pid_t		pid;

#endif /* SLAPD_COMMON_H */
