/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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
 * A copy of this license is available in the file LICENSE in the
 * top-level directory of the distribution or, alternatively, at
 * <http://www.OpenLDAP.org/license.html>.
 */

#include <unistd.h>

#include <lber.h>
#include <lber_pvt.h>
#include "lutil.h"
#include "lutil_md5.h"
#include <ac/string.h>

static LUTIL_PASSWD_CHK_FUNC chk_ns_mta_md5;
static const struct berval scheme = BER_BVC("{NS-MTA-MD5}");

#define NS_MTA_MD5_PASSLEN	64
static int chk_ns_mta_md5(
	const struct berval *scheme,
	const struct berval *passwd,
	const struct berval *cred,
	const char **text )
{
	lutil_MD5_CTX MD5context;
	unsigned char MD5digest[LUTIL_MD5_BYTES], c;
	char buffer[LUTIL_MD5_BYTES*2];
	int i;

	if( passwd->bv_len != NS_MTA_MD5_PASSLEN ) {
		return LUTIL_PASSWD_ERR;
	}

	/* hash credentials with salt */
	lutil_MD5Init(&MD5context);
	lutil_MD5Update(&MD5context,
		(const unsigned char *) &passwd->bv_val[32],
		32 );

	c = 0x59;
	lutil_MD5Update(&MD5context,
		(const unsigned char *) &c,
		1 );

	lutil_MD5Update(&MD5context,
		(const unsigned char *) cred->bv_val,
		cred->bv_len );

	c = 0xF7;
	lutil_MD5Update(&MD5context,
		(const unsigned char *) &c,
		1 );

	lutil_MD5Update(&MD5context,
		(const unsigned char *) &passwd->bv_val[32],
		32 );

	lutil_MD5Final(MD5digest, &MD5context);

	for( i=0; i < sizeof( MD5digest ); i++ ) {
		buffer[i+i]   = "0123456789abcdef"[(MD5digest[i]>>4) & 0x0F]; 
		buffer[i+i+1] = "0123456789abcdef"[ MD5digest[i] & 0x0F]; 
	}

	/* compare */
	return memcmp((char *)passwd->bv_val,
		(char *)buffer, sizeof(buffer)) ? LUTIL_PASSWD_ERR : LUTIL_PASSWD_OK;
}

int init_module(int argc, char *argv[]) {
	return lutil_passwd_add( (struct berval *)&scheme, chk_ns_mta_md5, NULL );
}
