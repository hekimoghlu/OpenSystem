/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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
/* $Id: serial.c,v 1.12 2007/06/19 23:47:17 tbox Exp $ */

/*! \file */

#include <config.h>

#include <isc/serial.h>

isc_boolean_t
isc_serial_lt(isc_uint32_t a, isc_uint32_t b) {
	/*
	 * Undefined => ISC_FALSE
	 */
	if (a == (b ^ 0x80000000U))
		return (ISC_FALSE);
	return (((isc_int32_t)(a - b) < 0) ? ISC_TRUE : ISC_FALSE);
}

isc_boolean_t
isc_serial_gt(isc_uint32_t a, isc_uint32_t b) {
	return (((isc_int32_t)(a - b) > 0) ? ISC_TRUE : ISC_FALSE);
}

isc_boolean_t
isc_serial_le(isc_uint32_t a, isc_uint32_t b) {
	return ((a == b) ? ISC_TRUE : isc_serial_lt(a, b));
}

isc_boolean_t
isc_serial_ge(isc_uint32_t a, isc_uint32_t b) {
	return ((a == b) ? ISC_TRUE : isc_serial_gt(a, b));
}

isc_boolean_t
isc_serial_eq(isc_uint32_t a, isc_uint32_t b) {
	return ((a == b) ? ISC_TRUE : ISC_FALSE);
}

isc_boolean_t
isc_serial_ne(isc_uint32_t a, isc_uint32_t b) {
	return ((a != b) ? ISC_TRUE : ISC_FALSE);
}
