/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
static inline void *
ns(_value)(struct ns() *t, uint32_t i)
{
	return (void *)((uintptr_t)t->keys[i] - t->key_offset);
}

static inline void
ns(_clear)(struct ns() *t)
{
	free(t->keys);
	ns(_init)(t, t->key_offset);
}

void
ns(_init)(struct ns() *t, size_t offset)
{
	*t = (struct ns()){
		.grow_shift = TABLE_MINSHIFT,
		.key_offset = (uint16_t)offset,
	};
}

OS_NOINLINE
static void
ns(_rehash)(struct ns() *t, int direction)
{
	struct ns() old = *t;

	if (direction > 0) {
		t->size += (1 << t->grow_shift);
		if (t->size == ((uint32_t)8 << t->grow_shift)) {
			t->grow_shift++;
		}
	} else if (direction < 0) {
		if (t->grow_shift > TABLE_MINSHIFT) {
			t->grow_shift--;
		}
		t->size = roundup(t->size / 2, (1 << t->grow_shift));
	}

	t->count = 0;
	t->tombstones = 0;
	t->keys = calloc(t->size, sizeof(key_t *));
	if (t->keys == NULL) {
		NOTIFY_INTERNAL_CRASH(0, "Unable to grow table: registration leak?");
	}

	for (uint32_t i = 0; i < old.size; i++) {
		if (old.keys[i] == NULL || old.keys[i] == TABLE_TOMBSTONE) {
			continue;
		}

		ns(_insert)(t, old.keys[i]);
	}
	free(old.keys);
}

void *
ns(_find)(struct ns() *t, ckey_t key)
{
	if (t->count == 0) {
		return NULL;
	}

	uint32_t size = t->size, loop_limit = t->size;
	uint32_t i = key_hash(key) % size;

	for (;;) {
		if (os_unlikely(loop_limit-- == 0)) {
			NOTIFY_INTERNAL_CRASH(0, "Corrupt hash table");
		}
		if (t->keys[i] != TABLE_TOMBSTONE) {
			if (t->keys[i] == NULL) {
				return NULL;
			}
			if (key_equals(key, *t->keys[i])) {
				return ns(_value)(t, i);
			}
		}
		i = table_next(i, size);
	}
}

void
ns(_insert)(struct ns() *t, key_t *key)
{
	/*
	 * Our algorithm relies on having enough NULLS to end loops.
	 * Make sure their density is never below 25%.
	 *
	 * When it drops too low, if the ratio of tombstones is low,
	 * assume we're on a growth codepath.
	 *
	 * Else, we just rehash in place to prune tombstones.
	 */
	if (os_unlikely(t->count + t->tombstones >= 3 * t->size / 4)) {
		if (t->count >= 4 * t->tombstones) {
			ns(_rehash)(t, 1);
		} else {
			ns(_rehash)(t, 0);
		}
	}

	uint32_t size = t->size, loop_limit = t->size;
	uint32_t i = key_hash(*key) % size;

	for (;;) {
		if (os_unlikely(loop_limit-- == 0)) {
			NOTIFY_INTERNAL_CRASH(0, "Corrupt hash table");
		}
		if (t->keys[i] == NULL) {
			break;
		}
		if (t->keys[i] == TABLE_TOMBSTONE) {
			t->tombstones--;
			break;
		}
		i = table_next(i, size);
	}

	t->keys[i] = key;
	t->count++;
}

void
ns(_delete)(struct ns() *t, ckey_t key)
{
	if (t->count == 0) {
		return;
	}

	uint32_t size = t->size, loop_limit = t->size;
	uint32_t i = key_hash(key) % size;

	for (;;) {
		if (os_unlikely(loop_limit-- == 0)) {
			NOTIFY_INTERNAL_CRASH(0, "Corrupt hash table");
		}
		if (t->keys[i] != TABLE_TOMBSTONE) {
			if (t->keys[i] == NULL) {
				return;
			}
			if (key_equals(key, *t->keys[i])) {
				break;
			}
		}
		i = table_next(i, size);
	}

	t->keys[i] = TABLE_TOMBSTONE;
	t->tombstones++;
	t->count--;

	if (t->keys[table_next(i, size)] == NULL) {
		do {
			t->tombstones--;
			t->keys[i] = NULL;
			i = table_prev(i, size);
		} while (t->keys[i] == TABLE_TOMBSTONE);
	}

	if (t->count == 0) {
		/* if the table is empty, free all its resources */
		ns(_clear)(t);
	} else if (t->size >= TABLE_MINSIZE * 2 && t->count < t->size / 8) {
		/* if the table density drops below 12%, shrink it */
		ns(_rehash)(t, -1);
	}
}

void
ns(_foreach)(struct ns() *t, bool (^handler)(void *))
{
	for (uint32_t i = 0; i < t->size; i++) {
		if (t->keys[i] != NULL && t->keys[i] != TABLE_TOMBSTONE) {
			if (!handler(ns(_value)(t, i))) break;
		}
	}
}

#undef ns
#undef key_t
#undef ckey_t
#undef key_hash
#undef key_equals
#undef make_map
