/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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
#ifndef __OS_COLLECTIONS_SET_H
#define __OS_COLLECTIONS_SET_H

#include <os/collections_set.h>

OS_ASSUME_NONNULL_BEGIN
__BEGIN_DECLS

#ifndef os_set_32_ptr_payload_handler_t
typedef bool (^os_set_32_ptr_payload_handler_t) (uint32_t *);
#endif

#ifndef os_set_64_ptr_payload_handler_t
typedef bool (^os_set_64_ptr_payload_handler_t) (uint64_t *);
#endif

#ifndef os_set_str_ptr_payload_handler_t
typedef bool (^os_set_str_ptr_payload_handler_t) (const char **);
#endif

__END_DECLS
OS_ASSUME_NONNULL_END

#define IN_SET(PREFIX, SUFFIX) PREFIX ## os_set_32_ptr ## SUFFIX
#define os_set_insert_val_t uint32_t *
#define os_set_find_val_t uint32_t
#include "_collections_set.in.h"
#undef IN_SET
#undef os_set_insert_val_t
#undef os_set_find_val_t

#define IN_SET(PREFIX, SUFFIX) PREFIX ## os_set_64_ptr ## SUFFIX
#define os_set_insert_val_t uint64_t *
#define os_set_find_val_t uint64_t
#include "_collections_set.in.h"
#undef IN_SET
#undef os_set_insert_val_t
#undef os_set_find_val_t

#define IN_SET(PREFIX, SUFFIX) PREFIX ## os_set_str_ptr ## SUFFIX
#define os_set_insert_val_t const char **
#define os_set_find_val_t const char *
#include "_collections_set.in.h"
#undef IN_SET
#undef os_set_insert_val_t
#undef os_set_find_val_t

#endif
