/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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
/* $Id: condition.h,v 1.6 2007/06/19 23:47:18 tbox Exp $ */

/*
 * This provides a limited subset of the isc_condition_t
 * functionality for use by single-threaded programs that
 * need to block waiting for events.   Only a single
 * call to isc_condition_wait() may be blocked at any given
 * time, and the _waituntil and _broadcast functions are not
 * supported.  This is intended primarily for use by the omapi
 * library, and may go away once omapi goes away.  Use for
 * other purposes is strongly discouraged.
 */

#ifndef ISC_CONDITION_H
#define ISC_CONDITION_H 1

#include <isc/mutex.h>

typedef int isc_condition_t;

isc_result_t isc__nothread_wait_hack(isc_condition_t *cp, isc_mutex_t *mp);
isc_result_t isc__nothread_signal_hack(isc_condition_t *cp);

#define isc_condition_init(cp) \
	(*(cp) = 0, ISC_R_SUCCESS)

#define isc_condition_wait(cp, mp) \
	isc__nothread_wait_hack(cp, mp)

#define isc_condition_waituntil(cp, mp, tp) \
	((void)(cp), (void)(mp), (void)(tp), ISC_R_NOTIMPLEMENTED)

#define isc_condition_signal(cp) \
	isc__nothread_signal_hack(cp)

#define isc_condition_broadcast(cp) \
	((void)(cp), ISC_R_NOTIMPLEMENTED)

#define isc_condition_destroy(cp) \
	(*(cp) == 0 ? (*(cp) = -1, ISC_R_SUCCESS) : ISC_R_UNEXPECTED)

#endif /* ISC_CONDITION_H */
