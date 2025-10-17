/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
/* $Id: parseint.h,v 1.9 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_PARSEINT_H
#define ISC_PARSEINT_H 1

#include <isc/lang.h>
#include <isc/types.h>

/*! \file isc/parseint.h
 * \brief Parse integers, in a saner way than atoi() or strtoul() do.
 */

/***
 ***	Functions
 ***/

ISC_LANG_BEGINDECLS

isc_result_t
isc_parse_uint32(isc_uint32_t *uip, const char *string, int base);

isc_result_t
isc_parse_uint16(isc_uint16_t *uip, const char *string, int base);

isc_result_t
isc_parse_uint8(isc_uint8_t *uip, const char *string, int base);
/*%<
 * Parse the null-terminated string 'string' containing a base 'base'
 * integer, storing the result in '*uip'.  
 * The base is interpreted
 * as in strtoul().  Unlike strtoul(), leading whitespace, minus or
 * plus signs are not accepted, and all errors (including overflow)
 * are reported uniformly through the return value.
 *
 * Requires:
 *\li	'string' points to a null-terminated string
 *\li	0 <= 'base' <= 36
 *
 * Returns:
 *\li	#ISC_R_SUCCESS
 *\li	#ISC_R_BADNUMBER   The string is not numeric (in the given base)
 *\li	#ISC_R_RANGE	  The number is not representable as the requested type.
 */

ISC_LANG_ENDDECLS

#endif /* ISC_PARSEINT_H */
