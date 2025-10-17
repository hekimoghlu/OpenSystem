/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
#include <TargetConditionals.h>
#include <stddef.h>
#include <stdint.h>
#include <os/tsd.h>

/*
 * cerror takes the return value of the syscall, being non-zero, and
 * stores it in errno. It needs to return -1 to indicate failure but
 * 64-bit platforms need to ensure that possible 128-bit wide return
 * values are also properly set.
 */
#ifdef __LP64__
typedef unsigned __int128 cerror_return_t;
#else
typedef uint64_t cerror_return_t;
#endif

extern void _pthread_exit_if_canceled(int error);

#undef errno
int errno;

int *
__error(void)
{
	void *ptr = _os_tsd_get_direct(__TSD_ERRNO);
	if (ptr != NULL) {
		return (int*)ptr;
	}
	return &errno;
}

__attribute__((noinline))
cerror_return_t
cerror_nocancel(int err)
{
	errno = err;
	int *tsderrno = (int*)_os_tsd_get_direct(__TSD_ERRNO);
	if (tsderrno) {
		*tsderrno = err;
	}
	return -1;
}

__attribute__((noinline))
cerror_return_t
cerror(int err)
{
	_pthread_exit_if_canceled(err);
	return cerror_nocancel(err);
}
