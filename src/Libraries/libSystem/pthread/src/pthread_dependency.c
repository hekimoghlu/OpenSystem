/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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
#include "resolver.h"
#include "internal.h"

#define PREREQUISITE_FULFILLED  (~0u)

#if !VARIANT_DYLD
void
pthread_dependency_init_np(pthread_dependency_t *pr, pthread_t pth,
		pthread_dependency_attr_t *attrs)
{
	if (attrs) *(volatile char *)attrs;
	*pr = (pthread_dependency_t)PTHREAD_DEPENDENCY_INITIALIZER_NP(pth);
}

OS_NOINLINE
void
_pthread_dependency_fulfill_slow(pthread_dependency_t *pr, uint32_t old)
{
	if (old == PREREQUISITE_FULFILLED) {
		PTHREAD_CLIENT_CRASH(0, "Fufilling pthread_dependency_t twice");
	}
	if (os_unlikely(old != _pthread_mach_thread_self_direct())) {
		PTHREAD_CLIENT_CRASH(old, "Fulfilled a dependency "
				"not owned by current thread");
	}

	int ret = __ulock_wake(UL_UNFAIR_LOCK | ULF_NO_ERRNO, &pr->__pdep_opaque1, 0);
	switch (-ret) {
	case 0:
	case ENOENT:
		return;
	default:
		PTHREAD_INTERNAL_CRASH(-ret, "__ulock_wake() failed");
	}
}


PTHREAD_NOEXPORT_VARIANT
void
pthread_dependency_fulfill_np(pthread_dependency_t *pr, void *value)
{
	uint32_t old;

	pr->__pdep_opaque2 = (uint64_t)(uintptr_t)value;
	old = os_atomic_xchg(&pr->__pdep_opaque1, PREREQUISITE_FULFILLED, release);

	if (old != 0) _pthread_dependency_fulfill_slow(pr, old);
}

PTHREAD_NOEXPORT_VARIANT
void *
pthread_dependency_wait_np(pthread_dependency_t *pr)
{
	if (os_atomic_cmpxchg(&pr->__pdep_opaque1, 0, pr->__pdep_owner, relaxed)) {
		int ret;
	again:
		ret = __ulock_wait(UL_UNFAIR_LOCK | ULF_NO_ERRNO, &pr->__pdep_opaque1,
				pr->__pdep_owner, 0);
		switch (-ret) {
		case EFAULT:
		case EINTR:
		case 0:
			if (pr->__pdep_opaque1 == pr->__pdep_owner) goto again;
			break;
		case EOWNERDEAD:
			PTHREAD_CLIENT_CRASH(pr->__pdep_owner, "Waiting on orphaned dependency");
		default:
			PTHREAD_CLIENT_CRASH(-ret, "__ulock_wait() failed");
		}
	}

	uint32_t cur = os_atomic_load(&pr->__pdep_opaque1, acquire);
	if (cur == PREREQUISITE_FULFILLED) {
		return (void *)(uintptr_t)pr->__pdep_opaque2;
	}
	PTHREAD_CLIENT_CRASH(cur, "Corrupted pthread_dependency_t");
}

PTHREAD_NOEXPORT_VARIANT void*
_pthread_atomic_xchg_ptr(void **p, void *v)
{
	return os_atomic_xchg(p, v, seq_cst);
}

PTHREAD_NOEXPORT_VARIANT uint32_t
_pthread_atomic_xchg_uint32_relaxed(uint32_t *p, uint32_t v)
{
	return os_atomic_xchg(p, v, relaxed);
}

#endif // !VARIANT_DYLD
