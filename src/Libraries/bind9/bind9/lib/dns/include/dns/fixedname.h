/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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
/* $Id: fixedname.h,v 1.19 2007/06/19 23:47:16 tbox Exp $ */

#ifndef DNS_FIXEDNAME_H
#define DNS_FIXEDNAME_H 1

/*****
 ***** Module Info
 *****/

/*! \file dns/fixedname.h
 * \brief
 * Fixed-size Names
 *
 * dns_fixedname_t is a convenience type containing a name, an offsets table,
 * and a dedicated buffer big enough for the longest possible name.
 *
 * MP:
 *\li	The caller must ensure any required synchronization.
 *
 * Reliability:
 *\li	No anticipated impact.
 *
 * Resources:
 *\li	Per dns_fixedname_t:
 *\code
 *		sizeof(dns_name_t) + sizeof(dns_offsets_t) +
 *		sizeof(isc_buffer_t) + 255 bytes + structure padding
 *\endcode
 *
 * Security:
 *\li	No anticipated impact.
 *
 * Standards:
 *\li	None.
 */

/*****
 ***** Imports
 *****/

#include <isc/buffer.h>

#include <dns/name.h>

/*****
 ***** Types
 *****/

struct dns_fixedname {
	dns_name_t			name;
	dns_offsets_t			offsets;
	isc_buffer_t			buffer;
	unsigned char			data[DNS_NAME_MAXWIRE];
};

#define dns_fixedname_init(fn) \
	do { \
		dns_name_init(&((fn)->name), (fn)->offsets); \
		isc_buffer_init(&((fn)->buffer), (fn)->data, \
                                  DNS_NAME_MAXWIRE); \
		dns_name_setbuffer(&((fn)->name), &((fn)->buffer)); \
	} while (0)

#define dns_fixedname_invalidate(fn) \
	dns_name_invalidate(&((fn)->name))

#define dns_fixedname_name(fn)		(&((fn)->name))

#endif /* DNS_FIXEDNAME_H */
