/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 24, 2023.
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
#ifndef _NOTIFY_TABLE_H_
#define _NOTIFY_TABLE_H_

#include <os/base.h>
#include <stdint.h>
#include <stdbool.h>

#include <os/collections.h>

struct _nc_table_ns {
	os_set_str_ptr_t set;
	size_t key_offset;
};

struct _nc_table_n_ns {
	os_set_32_ptr_t set;
	size_t key_offset;
};

struct _nc_table_64_ns {
	os_set_64_ptr_t set;
	size_t key_offset;
};

typedef struct _nc_table_ns table_t;
typedef struct _nc_table_n_ns table_n_t;
typedef struct _nc_table_64_ns table_64_t;

__BEGIN_DECLS

extern void _nc_table_init(table_t *t, size_t key_offset);
extern void _nc_table_init_n(table_n_t *t, size_t key_offset);
extern void _nc_table_init_64(table_64_t *t, size_t key_offset);

extern void _nc_table_insert(table_t *t, char **key);
extern void _nc_table_insert_n(table_n_t *t, uint32_t *key);
extern void _nc_table_insert_64(table_64_t *t, uint64_t *key);

extern void *_nc_table_find(table_t *t, const char *key);
extern void *_nc_table_find_n(table_n_t *t, uint32_t key);
extern void *_nc_table_find_64(table_64_t *t, uint64_t key);

extern void _nc_table_delete(table_t *t, const char *key, char **expected);
extern void _nc_table_delete_n(table_n_t *t, uint32_t key, uint32_t *expected);
extern void _nc_table_delete_64(table_64_t *t, uint64_t key, uint64_t *expected);

typedef bool (^payload_handler_t) (void *);

extern void _nc_table_foreach(table_t *t, OS_NOESCAPE payload_handler_t handler);
extern void _nc_table_foreach_n(table_n_t *t, OS_NOESCAPE payload_handler_t handler);
extern void _nc_table_foreach_64(table_64_t *t,OS_NOESCAPE payload_handler_t handler);

__END_DECLS

#endif /* _NOTIFY_TABLE_H_ */
