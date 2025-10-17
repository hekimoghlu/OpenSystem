/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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
#pragma once

#include <libc_hooks.h>

typedef struct {
    void *hook;
    const void *ptr;
    size_t size;
} libc_hooks_log_entry_t;
#define LIBC_HOOKS_LOG(hook, ptr, size) (libc_hooks_log_entry_t){(void *)hook, ptr, size}

typedef size_t libc_hooks_log_entry_idx_t;
static const libc_hooks_log_entry_idx_t libc_hooks_log_entry_max = 32;

typedef struct {
    libc_hooks_log_entry_idx_t count;
    libc_hooks_log_entry_idx_t check;
    libc_hooks_log_entry_t entries[libc_hooks_log_entry_max];
} libc_hooks_log_t;

static libc_hooks_log_t libc_hooks_log = {.count = 0, .check = 0};

static void libc_hooks_log_expect(libc_hooks_log_entry_t expected, const char *msg) {
    T_ASSERT_LT_UINT(libc_hooks_log.check, libc_hooks_log.count, "checking that there is room for another log entry");

    libc_hooks_log_entry_t entry = libc_hooks_log.entries[libc_hooks_log.check++];

    T_EXPECT_EQ_PTR(entry.hook, expected.hook, "%s - hook", msg);
    T_EXPECT_EQ_PTR(entry.ptr, expected.ptr, "%s - ptr", msg);
    T_EXPECT_EQ_ULONG(entry.size, expected.size, "%s - size", msg);
}

// These are the testing hooks. Their function is to append a record of being called to the in-memory log
static void libc_hooks_will_read(const void *p, size_t size) {
    assert(libc_hooks_log.count < libc_hooks_log_entry_max);
    libc_hooks_log.entries[libc_hooks_log.count++] = LIBC_HOOKS_LOG(libc_hooks_will_read, p, size);
}
static void libc_hooks_will_read_cstring(const char *s) {
    assert(libc_hooks_log.count < libc_hooks_log_entry_max - 1);
    libc_hooks_log.entries[libc_hooks_log.count++] = LIBC_HOOKS_LOG(libc_hooks_will_read_cstring, s, strlen(s) + 1);
}
static void libc_hooks_will_read_wcstring(const wchar_t *wcs) {
    assert(libc_hooks_log.count < libc_hooks_log_entry_max - 1);
    libc_hooks_log.entries[libc_hooks_log.count++] = LIBC_HOOKS_LOG((void *)libc_hooks_will_read_wcstring, wcs, wcslen(wcs) + 1 );
}
static void libc_hooks_will_write(const void *p, size_t size) {
    assert(libc_hooks_log.count < libc_hooks_log_entry_max - 1);
    libc_hooks_log.entries[libc_hooks_log.count++] = LIBC_HOOKS_LOG((void *)libc_hooks_will_write, p, size);
}

static void libc_hooks_log_start(void) {
    static libc_hooks_t libc_hooks_logging = {
        libc_hooks_version,
        libc_hooks_will_read,
        libc_hooks_will_read_cstring,
        libc_hooks_will_read_wcstring,
        libc_hooks_will_write
    };

    libc_hooks_log.check = libc_hooks_log.count = 0;
    libc_set_introspection_hooks(&libc_hooks_logging, NULL, sizeof(libc_hooks_t));
}

static void libc_hooks_log_stop(libc_hooks_log_entry_idx_t expected) {
    static libc_hooks_t libc_hooks_none = {libc_hooks_version, NULL, NULL, NULL, NULL};
    
    libc_set_introspection_hooks(&libc_hooks_none, NULL, sizeof(libc_hooks_t));
    libc_hooks_log.check = 0;

    T_EXPECT_EQ(libc_hooks_log.count, expected, "Checking number of log entries");
}

static void libc_hooks_log_dump(libc_hooks_log_t log) {
    printf("libc_hooks_will_read:          %p\n", (void *)libc_hooks_will_read);
    printf("libc_hooks_will_read_cstring:  %p\n", (void *)libc_hooks_will_read_cstring);
    printf("libc_hooks_will_read_wcstring: %p\n", (void *)libc_hooks_will_read_wcstring);
    printf("libc_hooks_will_write:         %p\n", (void *)libc_hooks_will_write);

    printf("log.check: %lu\n", log.check);
    printf("log.count: %lu\n", log.count);
    for (libc_hooks_log_entry_idx_t e = 0; e < log.count; e++)
        printf("log.entries[%u]: (%p, %p, %zu)\n", e, log.entries[e].hook, log.entries[e].ptr, log.entries[e].size);
}

// Locale helper:
//   locale_t is opaque (forward struct pointer) for consumers of the API (e.g. darwintests)
//   and so the size of the fully elaborated type (*locale_t) isn't available. We will have
//   to define that size by observing it empirically and recording it here so that we can
//   have an expectation for how much of a locale_t is being introspected. Observations are
//   recorded here:
#if TARGET_OS_WATCH && __ILP32__
#define SIZE_LOCALE_T 1400
#else
#define SIZE_LOCALE_T 1472
#endif
