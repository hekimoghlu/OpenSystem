/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
/*
 * Copyright 2008 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef _CTFMERGE_H
#define	_CTFMERGE_H

/*
 * Merging structures used in ctfmerge.  See ctfmerge.c for locking semantics.
 */

#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "ctftools.h"
#include "barrier.h"
#include "fifo.h"

typedef struct wip {
	pthread_mutex_t wip_lock;
	pthread_cond_t wip_cv;
	tdata_t *wip_td;
	int wip_nmerged;
	int wip_batchid;
} wip_t;

typedef struct workqueue {
	int wq_next_batchid;

	int wq_maxbatchsz;

	wip_t *wq_wip;
	int wq_nwipslots;
	int wq_nthreads;
	int wq_ithrottle;

	pthread_mutex_t wq_queue_lock;
	fifo_t *wq_queue;
	pthread_cond_t wq_work_avail;
	pthread_cond_t wq_work_removed;
	int wq_ninqueue;
	int wq_nextpownum;

	pthread_mutex_t wq_donequeue_lock;
	fifo_t *wq_donequeue;
	int wq_lastdonebatch;
	pthread_cond_t wq_done_cv;

	pthread_cond_t wq_alldone_cv; /* protected by queue_lock */
	int wq_alldone;

	int wq_nomorefiles;

	pthread_t *wq_thread;

	barrier_t wq_bar1;
	barrier_t wq_bar2;
} workqueue_t;

#ifdef __cplusplus
}
#endif

#endif /* _CTFMERGE_H */
