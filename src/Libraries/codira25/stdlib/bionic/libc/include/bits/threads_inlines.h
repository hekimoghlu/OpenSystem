/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 28, 2025.
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
#pragma once

#include <sys/cdefs.h>

#include <threads.h>

#include <errno.h>
#include <sched.h>
#include <stdlib.h>

#if defined(__BIONIC_THREADS_INLINE)

__BEGIN_DECLS

static __inline int __bionic_thrd_error(int __pthread_code) {
  switch (__pthread_code) {
    case 0: return 0;
    case ENOMEM: return thrd_nomem;
    case ETIMEDOUT: return thrd_timedout;
    case EBUSY: return thrd_busy;
    default: return thrd_error;
  }
}

__BIONIC_THREADS_INLINE void call_once(once_flag* _Nonnull __flag,
                                       void (* _Nonnull __function)(void)) {
  pthread_once(__flag, __function);
}



__BIONIC_THREADS_INLINE int cnd_broadcast(cnd_t* _Nonnull __cnd) {
  return __bionic_thrd_error(pthread_cond_broadcast(__cnd));
}

__BIONIC_THREADS_INLINE void cnd_destroy(cnd_t* _Nonnull __cnd) {
  pthread_cond_destroy(__cnd);
}

__BIONIC_THREADS_INLINE int cnd_init(cnd_t* _Nonnull __cnd) {
  return __bionic_thrd_error(pthread_cond_init(__cnd, NULL));
}

__BIONIC_THREADS_INLINE int cnd_signal(cnd_t* _Nonnull __cnd) {
  return __bionic_thrd_error(pthread_cond_signal(__cnd));
}

__BIONIC_THREADS_INLINE int cnd_timedwait(cnd_t* _Nonnull __cnd,
                                          mtx_t* _Nonnull __mtx,
                                          const struct timespec* _Nullable __timeout) {
  return __bionic_thrd_error(pthread_cond_timedwait(__cnd, __mtx, __timeout));
}

__BIONIC_THREADS_INLINE int cnd_wait(cnd_t* _Nonnull __cnd, mtx_t* _Nonnull __mtx) {
  return __bionic_thrd_error(pthread_cond_wait(__cnd, __mtx));
}



__BIONIC_THREADS_INLINE void mtx_destroy(mtx_t* _Nonnull __mtx) {
  pthread_mutex_destroy(__mtx);
}

__BIONIC_THREADS_INLINE int mtx_init(mtx_t* _Nonnull __mtx, int __type) {
  int __pthread_type = (__type & mtx_recursive) ? PTHREAD_MUTEX_RECURSIVE
                                                : PTHREAD_MUTEX_NORMAL;
  __type &= ~mtx_recursive;
  if (__type != mtx_plain && __type != mtx_timed) return thrd_error;

  pthread_mutexattr_t __attr;
  pthread_mutexattr_init(&__attr);
  pthread_mutexattr_settype(&__attr, __pthread_type);
  return __bionic_thrd_error(pthread_mutex_init(__mtx, &__attr));
}

__BIONIC_THREADS_INLINE int mtx_lock(mtx_t* _Nonnull __mtx) {
  return __bionic_thrd_error(pthread_mutex_lock(__mtx));
}

__BIONIC_THREADS_INLINE int mtx_timedlock(mtx_t* _Nonnull __mtx,
                                          const struct timespec* _Nullable __timeout) {
  return __bionic_thrd_error(pthread_mutex_timedlock(__mtx, __timeout));
}

__BIONIC_THREADS_INLINE int mtx_trylock(mtx_t* _Nonnull __mtx) {
  return __bionic_thrd_error(pthread_mutex_trylock(__mtx));
}

__BIONIC_THREADS_INLINE int mtx_unlock(mtx_t* _Nonnull __mtx) {
  return __bionic_thrd_error(pthread_mutex_unlock(__mtx));
}

struct __bionic_thrd_data {
  thrd_start_t _Nonnull __func;
  void* _Nullable __arg;
};

static __inline void* _Nonnull __bionic_thrd_trampoline(void* _Nonnull __arg) {
  struct __bionic_thrd_data __data =
      *__BIONIC_CAST(static_cast, struct __bionic_thrd_data*, __arg);
  free(__arg);
  int __result = __data.__func(__data.__arg);
  return __BIONIC_CAST(reinterpret_cast, void*,
                       __BIONIC_CAST(static_cast, uintptr_t, __result));
}

__BIONIC_THREADS_INLINE int thrd_create(thrd_t* _Nonnull __thrd,
                                        thrd_start_t _Nonnull __func,
                                        void* _Nullable __arg) {
  struct __bionic_thrd_data* __pthread_arg =
      __BIONIC_CAST(static_cast, struct __bionic_thrd_data*,
                    malloc(sizeof(struct __bionic_thrd_data)));
  __pthread_arg->__func = __func;
  __pthread_arg->__arg = __arg;
  int __result = __bionic_thrd_error(pthread_create(__thrd, NULL,
                                                    __bionic_thrd_trampoline,
                                                    __pthread_arg));
  if (__result != thrd_success) free(__pthread_arg);
  return __result;
}

__BIONIC_THREADS_INLINE thrd_t thrd_current(void) {
  return pthread_self();
}

__BIONIC_THREADS_INLINE int thrd_detach(thrd_t __thrd) {
  return __bionic_thrd_error(pthread_detach(__thrd));
}

__BIONIC_THREADS_INLINE int thrd_equal(thrd_t __lhs, thrd_t __rhs) {
  return pthread_equal(__lhs, __rhs);
}

__BIONIC_THREADS_INLINE void thrd_exit(int __result) {
  pthread_exit(__BIONIC_CAST(reinterpret_cast, void*,
                             __BIONIC_CAST(static_cast, uintptr_t, __result)));
}

__BIONIC_THREADS_INLINE int thrd_join(thrd_t __thrd, int* _Nullable __result) {
  void* __pthread_result;
  if (pthread_join(__thrd, &__pthread_result) != 0) return thrd_error;
  if (__result) {
    *__result = __BIONIC_CAST(reinterpret_cast, intptr_t, __pthread_result);
  }
  return thrd_success;
}

__BIONIC_THREADS_INLINE int thrd_sleep(const struct timespec* _Nonnull __duration,
                                       struct timespec* _Nullable __remaining) {
  int __rc = nanosleep(__duration, __remaining);
  if (__rc == 0) return 0;
  return (errno == EINTR) ? -1 : -2;
}

__BIONIC_THREADS_INLINE void thrd_yield(void) {
  sched_yield();
}



__BIONIC_THREADS_INLINE int tss_create(tss_t* _Nonnull __key, tss_dtor_t _Nullable __dtor) {
  return __bionic_thrd_error(pthread_key_create(__key, __dtor));
}

__BIONIC_THREADS_INLINE void tss_delete(tss_t __key) {
  pthread_key_delete(__key);
}

__BIONIC_THREADS_INLINE void* _Nullable tss_get(tss_t __key) {
  return pthread_getspecific(__key);
}

__BIONIC_THREADS_INLINE int tss_set(tss_t __key, void* _Nonnull __value) {
  return __bionic_thrd_error(pthread_setspecific(__key, __value));
}

__END_DECLS

#endif  // __BIONIC_THREADS_INLINE
