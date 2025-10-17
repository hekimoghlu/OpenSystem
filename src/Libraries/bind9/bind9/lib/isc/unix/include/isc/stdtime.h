/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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

#ifndef ISC_STDTIME_H
#define ISC_STDTIME_H 1

/*! \file */

#include <isc/lang.h>
#include <isc/int.h>

/*%
 * It's public information that 'isc_stdtime_t' is an unsigned integral type.
 * Applications that want maximum portability should not assume anything
 * about its size.
 */
typedef isc_uint32_t isc_stdtime_t;

/* but this flag helps... */
#define STDTIME_ON_32BITS	1

/*
 * isc_stdtime32_t is a 32-bit version of isc_stdtime_t.  A variable of this
 * type should only be used as an opaque integer (e.g.,) to compare two
 * time values.
 */
typedef isc_uint32_t isc_stdtime32_t;

ISC_LANG_BEGINDECLS
/* */
void
isc_stdtime_get(isc_stdtime_t *t);
/*%<
 * Set 't' to the number of seconds since 00:00:00 UTC, January 1, 1970.
 *
 * Requires:
 *
 *\li	't' is a valid pointer.
 */

#define isc_stdtime_convert32(t, t32p) (*(t32p) = t)
/*
 * Convert the standard time to its 32-bit version.
 */

ISC_LANG_ENDDECLS

#endif /* ISC_STDTIME_H */
