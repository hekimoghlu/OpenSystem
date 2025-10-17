/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
#ifndef RUBY_THREAD_PTHREAD_H
#define RUBY_THREAD_PTHREAD_H

#ifdef HAVE_PTHREAD_NP_H
#include <pthread_np.h>
#endif

#define RB_NATIVETHREAD_LOCK_INIT PTHREAD_MUTEX_INITIALIZER
#define RB_NATIVETHREAD_COND_INIT PTHREAD_COND_INITIALIZER

typedef pthread_cond_t rb_nativethread_cond_t;

typedef struct native_thread_data_struct {
    union {
        struct list_node ubf;
        struct list_node gvl;
    } node;
#if defined(__GLIBC__) || defined(__FreeBSD__)
    union
#else
    /*
     * assume the platform condvars are badly implemented and have a
     * "memory" of which mutex they're associated with
     */
    struct
#endif
    {
        rb_nativethread_cond_t intr; /* th->interrupt_lock */
        rb_nativethread_cond_t gvlq; /* vm->gvl.lock */
    } cond;
} native_thread_data_t;

#undef except
#undef try
#undef leave
#undef finally

typedef struct rb_global_vm_lock_struct {
    /* fast path */
    const struct rb_thread_struct *owner;
    rb_nativethread_lock_t lock; /* AKA vm->gvl.lock */

    /*
     * slow path, protected by vm->gvl.lock
     * - @waitq - FIFO queue of threads waiting for GVL
     * - @timer - it handles timeslices for @owner.  It is any one thread
     *   in @waitq, there is no @timer if @waitq is empty, but always
     *   a @timer if @waitq has entries
     * - @timer_err tracks timeslice limit, the timeslice only resets
     *   when pthread_cond_timedwait returns ETIMEDOUT, so frequent
     *   switching between contended/uncontended GVL won't reset the
     *   timer.
     */
    struct list_head waitq; /* <=> native_thread_data_t.node.ubf */
    const struct rb_thread_struct *timer;
    int timer_err;

    /* yield */
    rb_nativethread_cond_t switch_cond;
    rb_nativethread_cond_t switch_wait_cond;
    int need_yield;
    int wait_yield;
} rb_global_vm_lock_t;

#endif /* RUBY_THREAD_PTHREAD_H */
