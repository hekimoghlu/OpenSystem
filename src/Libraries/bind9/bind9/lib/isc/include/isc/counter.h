/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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
#ifndef ISC_COUNTER_H
#define ISC_COUNTER_H 1

/*****
 ***** Module Info
 *****/

/*! \file isc/counter.h
 *
 * \brief The isc_counter_t object is a simplified version of the
 * isc_quota_t object; it tracks the consumption of limited
 * resources, returning an error condition when the quota is
 * exceeded.  However, unlike isc_quota_t, attaching and detaching
 * from a counter object does not increment or decrement the counter.
 */

/***
 *** Imports.
 ***/

#include <isc/lang.h>
#include <isc/mutex.h>
#include <isc/types.h>

/*****
 ***** Types.
 *****/

ISC_LANG_BEGINDECLS

isc_result_t
isc_counter_create(isc_mem_t *mctx, int limit, isc_counter_t **counterp);
/*%<
 * Allocate and initialize a counter object.
 */

isc_result_t
isc_counter_increment(isc_counter_t *counter);
/*%<
 * Increment the counter.
 *
 * If the counter limit is nonzero and has been reached, then
 * return ISC_R_QUOTA, otherwise ISC_R_SUCCESS. (The counter is
 * incremented regardless of return value.)
 */

unsigned int
isc_counter_used(isc_counter_t *counter);
/*%<
 * Return the current counter value.
 */

void
isc_counter_setlimit(isc_counter_t *counter, int limit);
/*%<
 * Set the counter limit.
 */

void
isc_counter_attach(isc_counter_t *source, isc_counter_t **targetp);
/*%<
 * Attach to a counter object, increasing its reference counter.
 */

void
isc_counter_detach(isc_counter_t **counterp);
/*%<
 * Detach (and destroy if reference counter has dropped to zero)
 * a counter object.
 */

ISC_LANG_ENDDECLS

#endif /* ISC_COUNTER_H */
