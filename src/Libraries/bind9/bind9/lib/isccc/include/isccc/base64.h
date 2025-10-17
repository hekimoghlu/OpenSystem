/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
/* $Id: base64.h,v 1.10 2007/08/28 07:20:43 tbox Exp $ */

#ifndef ISCCC_BASE64_H
#define ISCCC_BASE64_H 1

/*! \file isccc/base64.h */

#include <isc/lang.h>
#include <isccc/types.h>

ISC_LANG_BEGINDECLS

/***
 *** Functions
 ***/

isc_result_t
isccc_base64_encode(isccc_region_t *source, int wordlength,
		  const char *wordbreak, isccc_region_t *target);
/*%<
 * Convert data into base64 encoded text.
 *
 * Notes:
 *\li	The base64 encoded text in 'target' will be divided into
 *	words of at most 'wordlength' characters, separated by
 * 	the 'wordbreak' string.  No parentheses will surround
 *	the text.
 *
 * Requires:
 *\li	'source' is a region containing binary data.
 *\li	'target' is a text region containing available space.
 *\li	'wordbreak' points to a null-terminated string of
 *		zero or more whitespace characters.
 */

isc_result_t
isccc_base64_decode(const char *cstr, isccc_region_t *target);
/*%<
 * Decode a null-terminated base64 string.
 *
 * Requires:
 *\li	'cstr' is non-null.
 *\li	'target' is a valid region.
 *
 * Returns:
 *\li	#ISC_R_SUCCESS	-- the entire decoded representation of 'cstring'
 *			   fit in 'target'.
 *\li	#ISC_R_BADBASE64 -- 'cstr' is not a valid base64 encoding.
 *\li	#ISC_R_NOSPACE	-- 'target' is not big enough.
 */

ISC_LANG_ENDDECLS

#endif /* ISCCC_BASE64_H */
