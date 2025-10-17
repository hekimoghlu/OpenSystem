/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
/* $Id: base64.h,v 1.22 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_BASE64_H
#define ISC_BASE64_H 1

/*! \file isc/base64.h */

#include <isc/lang.h>
#include <isc/types.h>

ISC_LANG_BEGINDECLS

/***
 *** Functions
 ***/

isc_result_t
isc_base64_totext(isc_region_t *source, int wordlength,
		  const char *wordbreak, isc_buffer_t *target);
/*!<
 * \brief Convert data into base64 encoded text.
 *
 * Notes:
 *\li	The base64 encoded text in 'target' will be divided into
 *	words of at most 'wordlength' characters, separated by
 * 	the 'wordbreak' string.  No parentheses will surround
 *	the text.
 *
 * Requires:
 *\li	'source' is a region containing binary data
 *\li	'target' is a text buffer containing available space
 *\li	'wordbreak' points to a null-terminated string of
 *		zero or more whitespace characters
 *
 * Ensures:
 *\li	target will contain the base64 encoded version of the data
 *	in source.  The 'used' pointer in target will be advanced as
 *	necessary.
 */

isc_result_t
isc_base64_decodestring(const char *cstr, isc_buffer_t *target);
/*!<
 * \brief Decode a null-terminated base64 string.
 *
 * Requires:
 *\li	'cstr' is non-null.
 *\li	'target' is a valid buffer.
 *
 * Returns:
 *\li	#ISC_R_SUCCESS	-- the entire decoded representation of 'cstring'
 *			   fit in 'target'.
 *\li	#ISC_R_BADBASE64 -- 'cstr' is not a valid base64 encoding.
 *
 * 	Other error returns are any possible error code from:
 *\li		isc_lex_create(),
 *\li		isc_lex_openbuffer(),
 *\li		isc_base64_tobuffer().
 */

isc_result_t
isc_base64_tobuffer(isc_lex_t *lexer, isc_buffer_t *target, int length);
/*!<
 * \brief Convert base64 encoded text from a lexer context into data.
 *
 * Requires:
 *\li	'lex' is a valid lexer context
 *\li	'target' is a buffer containing binary data
 *\li	'length' is an integer
 *
 * Ensures:
 *\li	target will contain the data represented by the base64 encoded
 *	string parsed by the lexer.  No more than length bytes will be read,
 *	if length is positive.  The 'used' pointer in target will be
 *	advanced as necessary.
 */



ISC_LANG_ENDDECLS

#endif /* ISC_BASE64_H */
