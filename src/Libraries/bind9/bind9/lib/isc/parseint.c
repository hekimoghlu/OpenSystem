/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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
/* $Id: parseint.c,v 1.8 2007/06/19 23:47:17 tbox Exp $ */

/*! \file */

#include <config.h>

#include <ctype.h>
#include <errno.h>
#include <limits.h>

#include <isc/parseint.h>
#include <isc/result.h>
#include <isc/stdlib.h>

isc_result_t
isc_parse_uint32(isc_uint32_t *uip, const char *string, int base) {
	unsigned long n;
	isc_uint32_t r;
	char *e;
	if (! isalnum((unsigned char)(string[0])))
		return (ISC_R_BADNUMBER);
	errno = 0;
	n = strtoul(string, &e, base);
	if (*e != '\0')
		return (ISC_R_BADNUMBER);
	/*
	 * Where long is 64 bits we need to convert to 32 bits then test for
	 * equality.  This is a no-op on 32 bit machines and a good compiler
	 * will optimise it away.
	 */
	r = (isc_uint32_t)n;
	if ((n == ULONG_MAX && errno == ERANGE) || (n != (unsigned long)r))
		return (ISC_R_RANGE);
	*uip = r;
	return (ISC_R_SUCCESS);
}

isc_result_t
isc_parse_uint16(isc_uint16_t *uip, const char *string, int base) {
	isc_uint32_t val;
	isc_result_t result;
	result = isc_parse_uint32(&val, string, base);
	if (result != ISC_R_SUCCESS)
		return (result);
	if (val > 0xFFFF)
		return (ISC_R_RANGE);
	*uip = (isc_uint16_t) val;
	return (ISC_R_SUCCESS);
}

isc_result_t
isc_parse_uint8(isc_uint8_t *uip, const char *string, int base) {
	isc_uint32_t val;
	isc_result_t result;
	result = isc_parse_uint32(&val, string, base);
	if (result != ISC_R_SUCCESS)
		return (result);
	if (val > 0xFF)
		return (ISC_R_RANGE);
	*uip = (isc_uint8_t) val;
	return (ISC_R_SUCCESS);
}
