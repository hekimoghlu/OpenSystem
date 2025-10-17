/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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
/* $Id: thread.h,v 1.26 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_THREAD_H
#define ISC_THREAD_H 1

/*! \file */

#include <pthread.h>

#if defined(HAVE_PTHREAD_NP_H)
#include <pthread_np.h>
#endif

#include <isc/lang.h>
#include <isc/result.h>

ISC_LANG_BEGINDECLS

typedef pthread_t isc_thread_t;
typedef void * isc_threadresult_t;
typedef void * isc_threadarg_t;
typedef isc_threadresult_t (*isc_threadfunc_t)(isc_threadarg_t);
typedef pthread_key_t isc_thread_key_t;

isc_result_t
isc_thread_create(isc_threadfunc_t, isc_threadarg_t, isc_thread_t *);

void
isc_thread_setconcurrency(unsigned int level);

void
isc_thread_yield(void);

void
isc_thread_setname(isc_thread_t thread, const char *name);

/* XXX We could do fancier error handling... */

#define isc_thread_join(t, rp) \
	((pthread_join((t), (rp)) == 0) ? \
	 ISC_R_SUCCESS : ISC_R_UNEXPECTED)

#define isc_thread_self \
	(unsigned long)pthread_self

#define isc_thread_key_create pthread_key_create
#define isc_thread_key_getspecific pthread_getspecific
#define isc_thread_key_setspecific pthread_setspecific
#define isc_thread_key_delete pthread_key_delete

ISC_LANG_ENDDECLS

#endif /* ISC_THREAD_H */
