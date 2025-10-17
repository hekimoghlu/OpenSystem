/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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

#ifndef _LUTIL_HASH_H_
#define _LUTIL_HASH_H_

#include <lber_types.h>

LDAP_BEGIN_DECL

#define LUTIL_HASH_BYTES 4

struct lutil_HASHContext {
	ber_uint_t hash;
};

LDAP_LUTIL_F( void )
lutil_HASHInit LDAP_P((
	struct lutil_HASHContext *context));

LDAP_LUTIL_F( void )
lutil_HASHUpdate LDAP_P((
	struct lutil_HASHContext *context,
	unsigned char const *buf,
	ber_len_t len));

LDAP_LUTIL_F( void )
lutil_HASHFinal LDAP_P((
	unsigned char digest[LUTIL_HASH_BYTES],
	struct lutil_HASHContext *context));

typedef struct lutil_HASHContext lutil_HASH_CTX;

LDAP_END_DECL

#endif /* _LUTIL_HASH_H_ */
