/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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
#include <pthread.h>

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

#include <async_safe/log.h>

#include "platform/bionic/page.h"
#include "private/ErrnoRestorer.h"
#include "private/bionic_defs.h"
#include "pthread_internal.h"

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_init(pthread_attr_t* attr) {
  attr->flags = 0;
  attr->stack_base = nullptr;
  attr->stack_size = PTHREAD_STACK_SIZE_DEFAULT;
  attr->guard_size = PTHREAD_GUARD_SIZE;
  attr->sched_policy = SCHED_NORMAL;
  attr->sched_priority = 0;
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_destroy(pthread_attr_t* attr) {
  memset(attr, 0xdf, sizeof(pthread_attr_t));
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_setinheritsched(pthread_attr_t* attr, int flag) {
  if (flag == PTHREAD_EXPLICIT_SCHED) {
    attr->flags &= ~PTHREAD_ATTR_FLAG_INHERIT;
    attr->flags |= PTHREAD_ATTR_FLAG_EXPLICIT;
  } else if (flag == PTHREAD_INHERIT_SCHED) {
    attr->flags |= PTHREAD_ATTR_FLAG_INHERIT;
    attr->flags &= ~PTHREAD_ATTR_FLAG_EXPLICIT;
  } else {
    return EINVAL;
  }
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_getinheritsched(const pthread_attr_t* attr, int* flag) {
  if ((attr->flags & PTHREAD_ATTR_FLAG_INHERIT) != 0) {
    *flag = PTHREAD_INHERIT_SCHED;
  } else if ((attr->flags & PTHREAD_ATTR_FLAG_EXPLICIT) != 0) {
    *flag = PTHREAD_EXPLICIT_SCHED;
  } else {
    // Historical behavior before P, when pthread_attr_setinheritsched was added.
    *flag = (attr->sched_policy != SCHED_NORMAL) ? PTHREAD_EXPLICIT_SCHED : PTHREAD_INHERIT_SCHED;
  }
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_setdetachstate(pthread_attr_t* attr, int state) {
  if (state == PTHREAD_CREATE_DETACHED) {
    attr->flags |= PTHREAD_ATTR_FLAG_DETACHED;
  } else if (state == PTHREAD_CREATE_JOINABLE) {
    attr->flags &= ~PTHREAD_ATTR_FLAG_DETACHED;
  } else {
    return EINVAL;
  }
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_getdetachstate(const pthread_attr_t* attr, int* state) {
  *state = (attr->flags & PTHREAD_ATTR_FLAG_DETACHED) ? PTHREAD_CREATE_DETACHED : PTHREAD_CREATE_JOINABLE;
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_setschedpolicy(pthread_attr_t* attr, int policy) {
  attr->sched_policy = policy;
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_getschedpolicy(const pthread_attr_t* attr, int* policy) {
  *policy = attr->sched_policy;
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_setschedparam(pthread_attr_t* attr, const sched_param* param) {
  attr->sched_priority = param->sched_priority;
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_getschedparam(const pthread_attr_t* attr, sched_param* param) {
  param->sched_priority = attr->sched_priority;
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_setstacksize(pthread_attr_t* attr, size_t stack_size) {
  if (stack_size < PTHREAD_STACK_MIN) {
    return EINVAL;
  }
  attr->stack_size = stack_size;
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_getstacksize(const pthread_attr_t* attr, size_t* stack_size) {
  void* unused;
  return pthread_attr_getstack(attr, &unused, stack_size);
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_setstack(pthread_attr_t* attr, void* stack_base, size_t stack_size) {
  if ((stack_size & (page_size() - 1) || stack_size < PTHREAD_STACK_MIN)) {
    return EINVAL;
  }
  if (reinterpret_cast<uintptr_t>(stack_base) & (page_size() - 1)) {
    return EINVAL;
  }
  attr->stack_base = stack_base;
  attr->stack_size = stack_size;
  return 0;
}

static int __pthread_attr_getstack_main_thread(void** stack_base, size_t* stack_size) {
  ErrnoRestorer errno_restorer;

  rlimit stack_limit;
  if (getrlimit(RLIMIT_STACK, &stack_limit) == -1) {
    return errno;
  }

  // If the current RLIMIT_STACK is RLIM_INFINITY, only admit to an 8MiB stack
  // in case callers such as ART take infinity too literally.
  if (stack_limit.rlim_cur == RLIM_INFINITY) {
    stack_limit.rlim_cur = 8 * 1024 * 1024;
  }
  uintptr_t lo, hi;
  __find_main_stack_limits(&lo, &hi);
  *stack_size = stack_limit.rlim_cur;
  *stack_base = reinterpret_cast<void*>(hi - *stack_size);
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_getstack(const pthread_attr_t* attr, void** stack_base, size_t* stack_size) {
  *stack_base = attr->stack_base;
  *stack_size = attr->stack_size;
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_setguardsize(pthread_attr_t* attr, size_t guard_size) {
  attr->guard_size = guard_size;
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_getguardsize(const pthread_attr_t* attr, size_t* guard_size) {
  *guard_size = attr->guard_size;
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_getattr_np(pthread_t t, pthread_attr_t* attr) {
  pthread_internal_t* thread = reinterpret_cast<pthread_internal_t*>(t);
  *attr = thread->attr;
  // We prefer reading join_state here to setting thread->attr.flags in pthread_detach.
  // Because data race exists in the latter case.
  if (atomic_load(&thread->join_state) == THREAD_DETACHED) {
    attr->flags |= PTHREAD_ATTR_FLAG_DETACHED;
  }
  // The main thread's stack information is not stored in thread->attr, and we need to
  // collect that at runtime.
  if (thread->tid == getpid()) {
    return __pthread_attr_getstack_main_thread(&attr->stack_base, &attr->stack_size);
  }
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_setscope(pthread_attr_t*, int scope) {
  if (scope == PTHREAD_SCOPE_SYSTEM) {
    return 0;
  }
  if (scope == PTHREAD_SCOPE_PROCESS) {
    return ENOTSUP;
  }
  return EINVAL;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_attr_getscope(const pthread_attr_t*, int* scope) {
  *scope = PTHREAD_SCOPE_SYSTEM;
  return 0;
}
