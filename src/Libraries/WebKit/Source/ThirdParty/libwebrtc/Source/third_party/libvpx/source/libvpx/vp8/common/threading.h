/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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
#ifndef VPX_VP8_COMMON_THREADING_H_
#define VPX_VP8_COMMON_THREADING_H_

#include "./vpx_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#if CONFIG_OS_SUPPORT && CONFIG_MULTITHREAD

#if defined(_WIN32) && !HAVE_PTHREAD_H
/* Win32 */
#include <windows.h>
#else
/* pthreads */
#ifdef __APPLE__
#include <mach/mach_init.h>
#include <mach/semaphore.h>
#include <mach/task.h>
#include <time.h>
#include <unistd.h>
#else
#include <semaphore.h>
#endif
#endif

/* Synchronization macros: Win32 and Pthreads */
#if defined(_WIN32) && !HAVE_PTHREAD_H
#define vp8_sem_t HANDLE
#define vp8_sem_init(sem, pshared, value) \
  (int)((*sem = CreateSemaphore(NULL, value, 32768, NULL)) == NULL)
#define vp8_sem_wait(sem) \
  (int)(WAIT_OBJECT_0 != WaitForSingleObject(*sem, INFINITE))
#define vp8_sem_post(sem) ReleaseSemaphore(*sem, 1, NULL)
#define vp8_sem_destroy(sem) \
  if (*sem) ((int)(CloseHandle(*sem)) == TRUE)
#define thread_sleep(nms) Sleep(nms)

#else

#ifdef __APPLE__
#define vp8_sem_t semaphore_t
#define vp8_sem_init(sem, pshared, value) \
  semaphore_create(mach_task_self(), sem, SYNC_POLICY_FIFO, value)
#define vp8_sem_wait(sem) semaphore_wait(*sem)
#define vp8_sem_post(sem) semaphore_signal(*sem)
#define vp8_sem_destroy(sem) semaphore_destroy(mach_task_self(), *sem)
#else
#include <errno.h>
#include <unistd.h>
#include <sched.h>
#define vp8_sem_t sem_t
#define vp8_sem_init sem_init
static INLINE int vp8_sem_wait(vp8_sem_t *sem) {
  int ret;
  while ((ret = sem_wait(sem)) == -1 && errno == EINTR) {
  }
  return ret;
}
#define vp8_sem_post sem_post
#define vp8_sem_destroy sem_destroy
#endif /* __APPLE__ */
/* Not Windows. Assume pthreads */

/* thread_sleep implementation: yield unless Linux/Unix. */
#if defined(__unix__) || defined(__APPLE__)
#define thread_sleep(nms)
/* {struct timespec ts;ts.tv_sec=0;
    ts.tv_nsec = 1000*nms;nanosleep(&ts, NULL);} */
#else
#define thread_sleep(nms) sched_yield();
#endif /* __unix__ || __APPLE__ */

#endif

#if VPX_ARCH_X86 || VPX_ARCH_X86_64
#include "vpx_ports/x86.h"
#else
#define x86_pause_hint()
#endif

#include "vpx_util/vpx_atomics.h"

static INLINE void vp8_atomic_spin_wait(
    int mb_col, const vpx_atomic_int *last_row_current_mb_col,
    const int nsync) {
  while (mb_col > (vpx_atomic_load_acquire(last_row_current_mb_col) - nsync)) {
    x86_pause_hint();
    thread_sleep(0);
  }
}

#endif /* CONFIG_OS_SUPPORT && CONFIG_MULTITHREAD */

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_THREADING_H_
