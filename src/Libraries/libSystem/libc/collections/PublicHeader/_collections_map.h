/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
#ifndef __OS_COLLECTIONS_MAP_H
#define __OS_COLLECTIONS_MAP_H

#include <os/collections_map.h>

OS_ASSUME_NONNULL_BEGIN
__BEGIN_DECLS

#ifndef os_map_str_payload_handler_t
typedef bool (^os_map_str_payload_handler_t) (const char *, void *);
#endif

#ifndef os_map_32_payload_handler_t
typedef bool (^os_map_32_payload_handler_t) (uint32_t, void *);
#endif

#ifndef os_map_64_payload_handler_t
typedef bool (^os_map_64_payload_handler_t) (uint64_t, void *);
#endif

#ifndef os_map_128_payload_handler_t
typedef bool (^os_map_128_payload_handler_t) (os_map_128_key_t, void *);
#endif

OS_EXPORT
const char *
os_map_str_entry(os_map_str_t *m, const char *key);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline const char * _Nullable
os_map_entry(os_map_str_t *m, const char *key)
{
	return os_map_str_entry(m, key);
}

__END_DECLS
OS_ASSUME_NONNULL_END

#define IN_MAP(PREFIX, SUFFIX) PREFIX ## os_map_str ## SUFFIX
#define os_map_key_t const char *
#include "_collections_map.in.h"
#undef IN_MAP
#undef os_map_key_t

#define IN_MAP(PREFIX, SUFFIX) PREFIX ## os_map_32 ## SUFFIX
#define os_map_key_t uint32_t
#include "_collections_map.in.h"
#undef IN_MAP
#undef os_map_key_t

#define IN_MAP(PREFIX, SUFFIX) PREFIX ## os_map_64 ## SUFFIX
#define os_map_key_t uint64_t
#include "_collections_map.in.h"
#undef IN_MAP
#undef os_map_key_t

#define IN_MAP(PREFIX, SUFFIX) PREFIX ## os_map_128 ## SUFFIX
#define os_map_key_t os_map_128_key_t
#include "_collections_map.in.h"
#undef IN_MAP
#undef os_map_key_t

#endif /* __OS_COLLECTIONS_MAP_H */
