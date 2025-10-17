/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 1, 2024.
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
/* $Id$ */

#ifndef _WIND_H_
#define _WIND_H_

#include <stddef.h>
#include <krb5-types.h>

#include <wind_err.h>

typedef unsigned int wind_profile_flags;

#define WIND_PROFILE_NAME 			0x00000001
#define WIND_PROFILE_SASL 			0x00000002
#define WIND_PROFILE_LDAP 			0x00000004
#define WIND_PROFILE_LDAP_CASE			0x00000008

#define WIND_PROFILE_LDAP_CASE_EXACT_ATTRIBUTE	0x00010000
#define WIND_PROFILE_LDAP_CASE_EXACT_ASSERTION	0x00020000
#define WIND_PROFILE_LDAP_NUMERIC		0x00040000
#define WIND_PROFILE_LDAP_TELEPHONE		0x00080000


/* flags to wind_ucs2read/wind_ucs2write */
#define WIND_RW_LE	1
#define WIND_RW_BE	2
#define WIND_RW_BOM	4

int wind_stringprep(const uint32_t *, size_t,
		    uint32_t *, size_t *,
		    wind_profile_flags);
int wind_profile(const char *, wind_profile_flags *);

int wind_punycode_label_toascii(const uint32_t *, size_t,
				char *, size_t *);

int wind_utf8ucs4(const char *, uint32_t *, size_t *);
int wind_utf8ucs4_copy(const char *, uint32_t **, size_t *);
int wind_utf8ucs4_length(const char *, size_t *);

int wind_ucs4utf8(const uint32_t *, size_t, char *, size_t *);
int wind_ucs4utf8_copy(const uint32_t *, size_t, char **, size_t *);
int wind_ucs4utf8_length(const uint32_t *, size_t, size_t *);

int wind_utf8ucs2(const char *, uint16_t *, size_t *);
int wind_utf8ucs2_length(const char *, size_t *);

int wind_ucs2utf8(const uint16_t *, size_t, char *, size_t *);
int wind_ucs2utf8_length(const uint16_t *, size_t, size_t *);


int wind_ucs2read(const void *, size_t, unsigned int *, uint16_t *, size_t *);
int wind_ucs2write(const uint16_t *, size_t, unsigned int *, void *, size_t *);

#endif /* _WIND_H_ */
