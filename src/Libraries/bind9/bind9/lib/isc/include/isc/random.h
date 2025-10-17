/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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
/* $Id: random.h,v 1.20 2009/01/17 23:47:43 tbox Exp $ */

#ifndef ISC_RANDOM_H
#define ISC_RANDOM_H 1

#include <isc/lang.h>
#include <isc/types.h>

/*! \file isc/random.h
 * \brief Implements a random state pool which will let the caller return a
 * series of possibly non-reproducible random values.
 *
 * Note that the
 * strength of these numbers is not all that high, and should not be
 * used in cryptography functions.  It is useful for jittering values
 * a bit here and there, such as timeouts, etc.
 */

ISC_LANG_BEGINDECLS

void
isc_random_seed(isc_uint32_t seed);
/*%<
 * Set the initial seed of the random state.
 */

void
isc_random_get(isc_uint32_t *val);
/*%<
 * Get a random value.
 *
 * Requires:
 *	val != NULL.
 */

isc_uint32_t
isc_random_jitter(isc_uint32_t max, isc_uint32_t jitter);
/*%<
 * Get a random value between (max - jitter) and (max).
 * This is useful for jittering timer values.
 */

ISC_LANG_ENDDECLS

#endif /* ISC_RANDOM_H */
