/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
#include <sys/types.h>
#include <sys/time.h>
#include <sys/errno.h>
#include <strings.h>
#include <Availability.h>

#include "gethostuuid_private.h"

extern int __gethostuuid(uuid_t, const struct timespec *);

static volatile int (*_gethostuuid_callback)(uuid_t) = (void *)0;

int
gethostuuid(uuid_t uuid, const struct timespec *timeout)
{
	int result;

	result = __gethostuuid(uuid, timeout);
	if ((result == -1) && (errno == EPERM)) {
		if (_gethostuuid_callback) {
			result = _gethostuuid_callback(uuid);
		} else {
			/* no fallback, return -1/EPERM */
			memset(uuid, 0x00, sizeof(*uuid));
		}
	}

	return result;
}

/* SPI to call gethostuuid syscall directly, without fallback, need an entitlement */
int
_getprivatesystemidentifier(uuid_t uuid, const struct timespec *timeout)
{
	return __gethostuuid(uuid, timeout);
}

int
_register_gethostuuid_callback(int (*new_callback)(uuid_t))
{
	if (__sync_bool_compare_and_swap((void **)&_gethostuuid_callback, (void *)0, (void *)new_callback)) {
		return 0;
	} else {
		return EINVAL;
	}
}
