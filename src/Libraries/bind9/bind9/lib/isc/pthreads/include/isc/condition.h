/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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
/* $Id: condition.h,v 1.26 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_CONDITION_H
#define ISC_CONDITION_H 1

/*! \file */

#include <isc/lang.h>
#include <isc/mutex.h>
#include <isc/result.h>
#include <isc/types.h>

typedef pthread_cond_t isc_condition_t;

#define isc_condition_init(cp) \
	((pthread_cond_init((cp), NULL) == 0) ? \
	 ISC_R_SUCCESS : ISC_R_UNEXPECTED)

#if ISC_MUTEX_PROFILE
#define isc_condition_wait(cp, mp) \
	((pthread_cond_wait((cp), &((mp)->mutex)) == 0) ? \
	 ISC_R_SUCCESS : ISC_R_UNEXPECTED)
#else
#define isc_condition_wait(cp, mp) \
	((pthread_cond_wait((cp), (mp)) == 0) ? \
	 ISC_R_SUCCESS : ISC_R_UNEXPECTED)
#endif

#define isc_condition_signal(cp) \
	((pthread_cond_signal((cp)) == 0) ? \
	 ISC_R_SUCCESS : ISC_R_UNEXPECTED)

#define isc_condition_broadcast(cp) \
	((pthread_cond_broadcast((cp)) == 0) ? \
	 ISC_R_SUCCESS : ISC_R_UNEXPECTED)

#define isc_condition_destroy(cp) \
	((pthread_cond_destroy((cp)) == 0) ? \
	 ISC_R_SUCCESS : ISC_R_UNEXPECTED)

ISC_LANG_BEGINDECLS

isc_result_t
isc_condition_waituntil(isc_condition_t *, isc_mutex_t *, isc_time_t *);

ISC_LANG_ENDDECLS

#endif /* ISC_CONDITION_H */
