/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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
/* $Id: serial.h,v 1.18 2009/01/18 23:48:14 tbox Exp $ */

#ifndef ISC_SERIAL_H
#define ISC_SERIAL_H 1

#include <isc/lang.h>
#include <isc/types.h>

/*! \file isc/serial.h
 *	\brief Implement 32 bit serial space arithmetic comparison functions.
 *	Note: Undefined results are returned as ISC_FALSE.
 */

/***
 ***	Functions
 ***/

ISC_LANG_BEGINDECLS

isc_boolean_t
isc_serial_lt(isc_uint32_t a, isc_uint32_t b);
/*%<
 *	Return true if 'a' < 'b' otherwise false.
 */

isc_boolean_t
isc_serial_gt(isc_uint32_t a, isc_uint32_t b);
/*%<
 *	Return true if 'a' > 'b' otherwise false.
 */

isc_boolean_t
isc_serial_le(isc_uint32_t a, isc_uint32_t b);
/*%<
 *	Return true if 'a' <= 'b' otherwise false.
 */

isc_boolean_t
isc_serial_ge(isc_uint32_t a, isc_uint32_t b);
/*%<
 *	Return true if 'a' >= 'b' otherwise false.
 */

isc_boolean_t
isc_serial_eq(isc_uint32_t a, isc_uint32_t b);
/*%<
 *	Return true if 'a' == 'b' otherwise false.
 */

isc_boolean_t
isc_serial_ne(isc_uint32_t a, isc_uint32_t b);
/*%<
 *	Return true if 'a' != 'b' otherwise false.
 */

ISC_LANG_ENDDECLS

#endif /* ISC_SERIAL_H */
