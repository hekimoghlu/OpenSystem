/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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
/* $Id: condition.h,v 1.17 2007/06/18 23:47:49 tbox Exp $ */

#ifndef ISC_CONDITION_H
#define ISC_CONDITION_H 1

#include <windows.h>

#include <isc/lang.h>
#include <isc/mutex.h>
#include <isc/thread.h>
#include <isc/types.h>

typedef struct isc_condition_thread isc_condition_thread_t;

struct isc_condition_thread {
	unsigned long				th;
	HANDLE					handle[2];
	ISC_LINK(isc_condition_thread_t)	link;

};

typedef struct isc_condition {
	HANDLE 		events[2];
	unsigned int	waiters;
	ISC_LIST(isc_condition_thread_t) threadlist;
} isc_condition_t;

ISC_LANG_BEGINDECLS

isc_result_t
isc_condition_init(isc_condition_t *);

isc_result_t
isc_condition_wait(isc_condition_t *, isc_mutex_t *);

isc_result_t
isc_condition_signal(isc_condition_t *);

isc_result_t
isc_condition_broadcast(isc_condition_t *);

isc_result_t
isc_condition_destroy(isc_condition_t *);

isc_result_t
isc_condition_waituntil(isc_condition_t *, isc_mutex_t *, isc_time_t *);

ISC_LANG_ENDDECLS

#endif /* ISC_CONDITION_H */
