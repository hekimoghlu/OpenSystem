/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
#include "libc_hooks.h"

#include <assert.h>

__attribute__((visibility("hidden"))) libc_hooks_t libc_hooks;

void
libc_set_introspection_hooks(const libc_hooks_t *new_hooks, libc_hooks_t *old_hooks, size_t size) {
	// Any version of the hooks always contains at least the version as the first field
	assert(size >= sizeof(libc_hooks_version));

	if (old_hooks) {
		// There are no older versions of the SPI so we can just assert here
		assert(size >= sizeof(libc_hooks));

		// If caller is offering an oversized libc_hooks_t, it could be
		// from a later version of the SPI zero that part out
		if (size > sizeof(libc_hooks))
			bzero(old_hooks + sizeof(libc_hooks), size - sizeof(libc_hooks));

		// We have the room to copy the current libc_hooks back to the user
		*old_hooks = libc_hooks;
	}

	if (new_hooks) {
		// There are no older versions of the SPI so we can just assert here
		assert(new_hooks->version >= libc_hooks_version);

		// The size had better be at least big enough to hold our libc_hooks
		// since libc_hooks is only allowed to grow
		assert(size >= sizeof(libc_hooks));

		// Copy new_hooks since it's lifetime of new_hooks is unknowable.
		libc_hooks = *new_hooks;

		// Set the version since we might have been offered a version of
		// libc_hooks_t from the future that we don't know what to do with.
		libc_hooks.version = libc_hooks_version;
	}
}
