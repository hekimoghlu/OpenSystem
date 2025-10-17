/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 2, 2025.
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

//
//  pthread.c
//  pthread
//
//  Created by Matt Wright on 9/13/12.
//  Copyright (c) 2012 Matt Wright. All rights reserved.
//

#include <kern/thread.h>
#include <kern/debug.h>
#include "kern/kern_internal.h"

kern_return_t pthread_start(kmod_info_t * ki, void *d);
kern_return_t pthread_stop(kmod_info_t *ki, void *d);

pthread_callbacks_t pthread_kern;

const struct pthread_functions_s pthread_internal_functions = {
	.pthread_init = _pthread_init,
	.pth_proc_hashinit = _pth_proc_hashinit,
	.pth_proc_hashdelete = _pth_proc_hashdelete,
	.bsdthread_create = _bsdthread_create,
	.bsdthread_register = _bsdthread_register,
	.bsdthread_terminate = _bsdthread_terminate,
	.thread_selfid = _thread_selfid,

	.psynch_mutexwait = _psynch_mutexwait,
	.psynch_mutexdrop = _psynch_mutexdrop,
	.psynch_cvbroad = _psynch_cvbroad,
	.psynch_cvsignal = _psynch_cvsignal,
	.psynch_cvwait = _psynch_cvwait,
	.psynch_cvclrprepost = _psynch_cvclrprepost,
	.psynch_rw_longrdlock = _psynch_rw_longrdlock,
	.psynch_rw_rdlock = _psynch_rw_rdlock,
	.psynch_rw_unlock = _psynch_rw_unlock,
	.psynch_rw_wrlock = _psynch_rw_wrlock,
	.psynch_rw_yieldwrlock = _psynch_rw_yieldwrlock,

	.pthread_find_owner = _pthread_find_owner,
	.pthread_get_thread_kwq = _pthread_get_thread_kwq,

	.workq_create_threadstack = workq_create_threadstack,
	.workq_destroy_threadstack = workq_destroy_threadstack,
	.workq_setup_thread = workq_setup_thread,
	.workq_handle_stack_events = workq_handle_stack_events,
	.workq_markfree_threadstack = workq_markfree_threadstack,
};

kern_return_t pthread_start(__unused kmod_info_t * ki, __unused void *d)
{
	pthread_kext_register((pthread_functions_t)&pthread_internal_functions, &pthread_kern);
	return KERN_SUCCESS;
}

kern_return_t pthread_stop(__unused kmod_info_t *ki, __unused void *d)
{
	return KERN_FAILURE;
}

struct uthread*
current_uthread(void)
{
	thread_t th = current_thread();
	return pthread_kern->get_bsdthread_info(th);
}
