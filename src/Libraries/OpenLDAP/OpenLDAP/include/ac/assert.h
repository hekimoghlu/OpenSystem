/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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

#ifndef _AC_ASSERT_H
#define _AC_ASSERT_H

#undef assert

#ifdef LDAP_DEBUG

#if defined( HAVE_ASSERT_H ) || defined( STDC_HEADERS )

#undef NDEBUG
#include <assert.h>

#else /* !(HAVE_ASSERT_H || STDC_HEADERS) */

#define LDAP_NEED_ASSERT 1

/*
 * no assert()... must be a very old compiler.
 * create a replacement and hope it works
 */

LBER_F (void) ber_pvt_assert LDAP_P(( const char *file, int line,
	const char *test ));

/* Can't use LDAP_STRING(test), that'd expand to "test" */
#if defined(__STDC__) || defined(__cplusplus)
#define assert(test) \
	((test) ? (void)0 : ber_pvt_assert( __FILE__, __LINE__, #test ) )
#else
#define assert(test) \
	((test) ? (void)0 : ber_pvt_assert( __FILE__, __LINE__, "test" ) )
#endif

#endif /* (HAVE_ASSERT_H || STDC_HEADERS) */

#else /* !LDAP_DEBUG */
/* no asserts */
#define assert(test) ((void)0)
#endif /* LDAP_DEBUG */

#endif /* _AC_ASSERT_H */
