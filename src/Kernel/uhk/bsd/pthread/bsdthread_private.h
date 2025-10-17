/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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
#ifndef _PTHREAD_BSDTHREAD_PRIVATE_H_
#define _PTHREAD_BSDTHREAD_PRIVATE_H_

#if XNU_KERNEL_PRIVATE && !defined(__PTHREAD_EXPOSE_INTERNALS__)
#define __PTHREAD_EXPOSE_INTERNALS__ 1
#endif // XNU_KERNEL_PRIVATE

#ifdef __PTHREAD_EXPOSE_INTERNALS__

/* pthread bsdthread_ctl sysctl commands */
/* bsdthread_ctl(BSDTHREAD_CTL_SET_QOS, thread_port, tsd_entry_addr, 0) */
#define BSDTHREAD_CTL_SET_QOS                           0x10
/* bsdthread_ctl(BSDTHREAD_CTL_GET_QOS, thread_port, 0, 0) */
#define BSDTHREAD_CTL_GET_QOS                           0x20
/* bsdthread_ctl(BSDTHREAD_CTL_QOS_OVERRIDE_START, thread_port, priority, 0) */
#define BSDTHREAD_CTL_QOS_OVERRIDE_START        0x40
/* bsdthread_ctl(BSDTHREAD_CTL_QOS_OVERRIDE_END, thread_port, 0, 0) */
#define BSDTHREAD_CTL_QOS_OVERRIDE_END          0x80
/* bsdthread_ctl(BSDTHREAD_CTL_SET_SELF, priority, voucher, flags) */
#define BSDTHREAD_CTL_SET_SELF                          0x100
/* bsdthread_ctl(BSDTHREAD_CTL_QOS_OVERRIDE_RESET, 0, 0, 0) */
#define BSDTHREAD_CTL_QOS_OVERRIDE_RESET        0x200
/* bsdthread_ctl(BSDTHREAD_CTL_QOS_OVERRIDE_DISPATCH, thread_port, priority, 0) */
#define BSDTHREAD_CTL_QOS_OVERRIDE_DISPATCH     0x400
/* bsdthread_ctl(BSDTHREAD_CTL_QOS_DISPATCH_ASYNCHRONOUS_OVERRIDE_ADD, thread_port, priority, resource) */
#define BSDTHREAD_CTL_QOS_DISPATCH_ASYNCHRONOUS_OVERRIDE_ADD            0x401
/* bsdthread_ctl(BSDTHREAD_CTL_QOS_DISPATCH_ASYNCHRONOUS_OVERRIDE_RESET, 0|1 (?reset_all), resource, 0) */
#define BSDTHREAD_CTL_QOS_DISPATCH_ASYNCHRONOUS_OVERRIDE_RESET          0x402
/* bsdthread_ctl(BSDTHREAD_CTL_QOS_MAX_PARALLELISM, priority, flags, 0) */
#define BSDTHREAD_CTL_QOS_MAX_PARALLELISM       0x800
/*
 * bsdthread_ctl(BSDTHREAD_CTL_WORKQ_ALLOW_KILL, enable, 0, 0)
 * It only affects the calling thread. Regular UNIX calls still need to be
 * used to manipulate signal mask of the calling thread to allow delivery
 * of a specific signal to it.
 * It is typically used in abort paths so it does not need to worry about
 * preserving sigmask across the thread's re-use. See workq_thread_return.
 */
#define BSDTHREAD_CTL_WORKQ_ALLOW_KILL 0x1000
/* bsdthread_ctl(BSDTHREAD_CTL_DISPATCH_APPLY_ATTR, flags, val1, val2) */
#define BSDTHREAD_CTL_DISPATCH_APPLY_ATTR 0x2000
/*
 * bsdthread_ctl(BSDTHREAD_CTL_WORKQ_ALLOW_SIGMASK, sigmask, 0, 0)
 * This is a process wide configuration (as opposed to ALLOW_KILL) that
 * provides the calling process an ability to send signals to all its
 * pthread workqueue threads.
 * Regular UNIX calls still need to be used to manipulate signal mask of
 * each individual pthread worker thread to allow delivery of a specific
 * signal to that thread.
 * The @sigmask specified here is used internally by workqueue subsystem
 * to preserve sigmask of pthread workqueue threads across their re-use.
 * See workq_thread_return.
 */

#define BSDTHREAD_CTL_WORKQ_ALLOW_SIGMASK 0x4000

/* Flags for BSDTHREAD_CTL_QOS_MAX_PARALLELISM */
#define _PTHREAD_QOS_PARALLELISM_COUNT_LOGICAL 0x1
#define _PTHREAD_QOS_PARALLELISM_REALTIME      0x2
#define _PTHREAD_QOS_PARALLELISM_CLUSTER_SHARED_RSRC      0x4


/* Flags for BSDTHREAD_CTL_DISPATCH_APPLY_ATTR */
#define _PTHREAD_DISPATCH_APPLY_ATTR_CLUSTER_SHARED_RSRC_SET    0x1
#define _PTHREAD_DISPATCH_APPLY_ATTR_CLUSTER_SHARED_RSRC_CLEAR  0x2


#endif // __PTHREAD_EXPOSE_INTERNALS__
#endif // _PTHREAD_BSDTHREAD_PRIVATE_H_
