/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 14, 2021.
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
#ifndef _OS_HASH_H_
#define _OS_HASH_H_
#if PRIVATE

#include <os/base.h>

__BEGIN_DECLS

static inline uint32_t
os_hash_jenkins_update(const void *data, size_t length, uint32_t hash)
{
	const uint8_t *key = (const uint8_t *)data;

	for (size_t i = 0; i < length; i++) {
		hash += key[i];
		hash += (hash << 10);
		hash ^= (hash >> 6);
	}

	return hash;
}

static inline uint32_t
os_hash_jenkins_finish(uint32_t hash)
{
	hash += (hash << 3);
	hash ^= (hash >> 11);
	hash += (hash << 15);

	return hash;
}

/*!
 * @function os_hash_jenkins
 *
 * @brief
 * The original Jenkins "one at a time" hash.
 *
 * @discussion
 * TBD: There may be some value to unrolling here,
 * depending on the architecture.
 *
 * @param data
 * The address of the data to hash.
 *
 * @param length
 * The length of the data to hash
 *
 * @param seed
 * An optional hash seed (defaults to 0).
 *
 * @returns
 * The jenkins hash for this data.
 */
__attribute__((overloadable))
static inline uint32_t
os_hash_jenkins(const void *data, size_t length, uint32_t seed)
{
	return os_hash_jenkins_finish(os_hash_jenkins_update(data, length, seed));
}

__attribute__((overloadable))
static inline uint32_t
os_hash_jenkins(const void *data, size_t length)
{
	return os_hash_jenkins(data, length, 0);
}

/*!
 * @function os_hash_kernel_pointer
 *
 * @brief
 * Hashes a pointer from a zone.
 *
 * @discussion
 * This is a really cheap and fast hash that will behave well for pointers
 * allocated by the kernel.
 *
 * This should be not used for untrusted pointer values from userspace,
 * or cases when the pointer is somehow under the control of userspace.
 *
 * This hash function utilizes knowledge about the span of the kernel
 * address space and inherent alignment of zalloc/kalloc.
 *
 * @param pointer
 * The pointer to hash.
 *
 * @returns
 * The hash for this pointer.
 */
static inline uint32_t
os_hash_kernel_pointer(const void *pointer)
{
	uintptr_t key = (uintptr_t)pointer >> 4;
	key *= 0x5052acdb;
	return (uint32_t)key ^ __builtin_bswap32((uint32_t)key);
}

__END_DECLS

#endif // PRIVATE
#endif // _OS_HASH_H_
