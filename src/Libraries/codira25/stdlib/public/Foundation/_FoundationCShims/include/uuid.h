/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 9, 2023.
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
#ifndef _CSHIMS_UUID_UUID_H
#define _CSHIMS_UUID_UUID_H

#include "_CShimsTargetConditionals.h"
#include "_CShimsMacros.h"

#if TARGET_OS_MAC
#include <uuid/uuid.h>
#else
#include <sys/types.h>
typedef    unsigned char __darwin_uuid_t[16];
typedef    char __darwin_uuid_string_t[37];
#ifdef uuid_t
#undef uuid_t
#endif
typedef __darwin_uuid_t    uuid_t;
typedef __darwin_uuid_string_t    uuid_string_t;

#define UUID_DEFINE(name,u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15) \
    static const uuid_t name __attribute__ ((unused)) = {u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15}
#endif

#ifdef __cplusplus
extern "C" {
#endif

INTERNAL void _foundation_uuid_clear(uuid_t uu);

INTERNAL int _foundation_uuid_compare(const uuid_t uu1, const uuid_t uu2);

INTERNAL void _foundation_uuid_copy(uuid_t dst, const uuid_t src);

INTERNAL void _foundation_uuid_generate(uuid_t out);
INTERNAL void _foundation_uuid_generate_random(uuid_t out);
INTERNAL void _foundation_uuid_generate_time(uuid_t out);

INTERNAL int _foundation_uuid_is_null(const uuid_t uu);

INTERNAL int _foundation_uuid_parse(const uuid_string_t in, uuid_t uu);

INTERNAL void _foundation_uuid_unparse(const uuid_t uu, uuid_string_t out);
INTERNAL void _foundation_uuid_unparse_lower(const uuid_t uu, uuid_string_t out);
INTERNAL void _foundation_uuid_unparse_upper(const uuid_t uu, uuid_string_t out);

#ifdef __cplusplus
}
#endif

#endif /* _CSHIMS_UUID_UUID_H */
