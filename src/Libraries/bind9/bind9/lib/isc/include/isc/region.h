/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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
/* $Id: region.h,v 1.25 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_REGION_H
#define ISC_REGION_H 1

/*! \file isc/region.h */

#include <isc/types.h>
#include <isc/lang.h>

struct isc_region {
	unsigned char *	base;
	unsigned int	length;
};

struct isc_textregion {
	char *		base;
	unsigned int	length;
};

/* XXXDCL questionable ... bears discussion.  we have been putting off
 * discussing the region api.
 */
struct isc_constregion {
	const void *	base;
	unsigned int	length;
};

struct isc_consttextregion {
	const char *	base;
	unsigned int	length;
};

/*@{*/
/*!
 * The region structure is not opaque, and is usually directly manipulated.
 * Some macros are defined below for convenience.
 */

#define isc_region_consume(r,l) \
	do { \
		isc_region_t *_r = (r); \
		unsigned int _l = (l); \
		INSIST(_r->length >= _l); \
		_r->base += _l; \
		_r->length -= _l; \
	} while (0)

#define isc_textregion_consume(r,l) \
	do { \
		isc_textregion_t *_r = (r); \
		unsigned int _l = (l); \
		INSIST(_r->length >= _l); \
		_r->base += _l; \
		_r->length -= _l; \
	} while (0)

#define isc_constregion_consume(r,l) \
	do { \
		isc_constregion_t *_r = (r); \
		unsigned int _l = (l); \
		INSIST(_r->length >= _l); \
		_r->base += _l; \
		_r->length -= _l; \
	} while (0)
/*@}*/

ISC_LANG_BEGINDECLS

int
isc_region_compare(isc_region_t *r1, isc_region_t *r2);
/*%<
 * Compares the contents of two regions
 *
 * Requires:
 *\li	'r1' is a valid region
 *\li	'r2' is a valid region
 *
 * Returns:
 *\li	 < 0 if r1 is lexicographically less than r2
 *\li	 = 0 if r1 is lexicographically identical to r2
 *\li	 > 0 if r1 is lexicographically greater than r2
 */

ISC_LANG_ENDDECLS

#endif /* ISC_REGION_H */
