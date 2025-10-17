/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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
#include <assert.h>

#include "table.h"
#include "notify_internal.h"

void _nc_table_init(table_t *t, size_t key_offset) {
	os_set_init(&t->set, NULL);
	t->key_offset = key_offset;
}

void _nc_table_init_n(table_n_t *t, size_t key_offset) {
	os_set_init(&t->set, NULL);
	t->key_offset = key_offset;
}

void _nc_table_init_64(table_64_t *t, size_t key_offset) {
	os_set_init(&t->set, NULL);
	t->key_offset = key_offset;
}

void _nc_table_insert(table_t *t, char **key) {
	os_set_insert(&t->set, (void *)key);
}

void _nc_table_insert_n(table_n_t *t, uint32_t *key)  {
	os_set_insert(&t->set, (void *)key);
}

void _nc_table_insert_64(table_64_t *t, uint64_t *key)  {
	os_set_insert(&t->set, (void *)key);
}

void *_nc_table_find(table_t *t, const char *key) {
	void *offset_result = os_set_find(&t->set, key);
	return (offset_result != NULL) ? (void *)((uintptr_t)offset_result - (uintptr_t)t->key_offset) : NULL;
}

void *_nc_table_find_n(table_n_t *t, uint32_t key) {
	void *offset_result = os_set_find(&t->set, key);
	return (offset_result != NULL) ? (void *)((uintptr_t)offset_result - (uintptr_t)t->key_offset)  : NULL;
}

void *_nc_table_find_64(table_64_t *t, uint64_t key) {
	void *offset_result = os_set_find(&t->set, key);
	return (offset_result != NULL) ? (void *)((uintptr_t)offset_result - (uintptr_t)t->key_offset)  : NULL;
}

void _nc_table_delete(table_t *t, const char *key, char **expected) {
	assert(os_set_delete(&t->set, key) == expected);
}

void _nc_table_delete_n(table_n_t *t, uint32_t key, uint32_t *expected) {
	assert(os_set_delete(&t->set, key) == expected);
}

void _nc_table_delete_64(table_64_t *t, uint64_t key, uint64_t *expected) {
	assert(os_set_delete(&t->set, key) == expected);
}

typedef bool (^payload_handler_t) (void *);

void _nc_table_foreach(table_t *t, OS_NOESCAPE payload_handler_t handler) {
	os_set_foreach(&t->set, ^bool (const char **_ptr) {
		return handler((void *)((uintptr_t)_ptr - (uintptr_t)t->key_offset));
	});
}

void _nc_table_foreach_n(table_n_t *t, OS_NOESCAPE payload_handler_t handler) {
	os_set_foreach(&t->set, ^bool (uint32_t *_ptr) {
		return handler((void *)((uintptr_t)_ptr - (uintptr_t)t->key_offset));
	});
}

void _nc_table_foreach_64(table_64_t *t,OS_NOESCAPE payload_handler_t handler) {
	os_set_foreach(&t->set, ^bool (uint64_t *_ptr) {
		return handler((void *)((uintptr_t)_ptr - (uintptr_t)t->key_offset));
	});
}
