/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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
 * A copy of this license is available in file LICENSE in the
 * top-level directory of the distribution or, alternatively, at
 * <http://www.OpenLDAP.org/license.html>.
 */

#ifndef _AC_ERRNO_H
#define _AC_ERRNO_H

#if defined( HAVE_ERRNO_H )
# include <errno.h>
#elif defined( HAVE_SYS_ERRNO_H )
# include <sys/errno.h>
#endif

#ifndef HAVE_SYS_ERRLIST
	/* no sys_errlist */
#	define		sys_nerr	0
#	define		sys_errlist	((char **)0)
#elif defined( DECL_SYS_ERRLIST )
	/* have sys_errlist but need declaration */
	LDAP_LIBC_V(int)      sys_nerr;
	LDAP_LIBC_V(char)    *sys_errlist[];
#endif

#undef _AC_ERRNO_UNKNOWN
#define _AC_ERRNO_UNKNOWN "unknown error"

#ifdef HAVE_SYS_ERRLIST
	/* this is thread safe */
#	define	STRERROR(e) ( (e) > -1 && (e) < sys_nerr \
			? sys_errlist[(e)] : _AC_ERRNO_UNKNOWN )

#elif defined( HAVE_STRERROR )
	/* this may not be thread safe */
	/* and, yes, some implementations of strerror may return NULL */
#	define	STRERROR(e) ( strerror(e) \
		? strerror(e) : _AC_ERRNO_UNKNOWN )

#else
	/* this is thread safe */
#	define	STRERROR(e) ( _AC_ERRNO_UNKNOWN )
#endif

#endif /* _AC_ERRNO_H */
