/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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

/**
 * @file android/crash_detail.h
 * @brief Attach extra information to android crashes.
 */

#include <sys/cdefs.h>

#include <stddef.h>

__BEGIN_DECLS

typedef struct crash_detail_t crash_detail_t;

/**
 * Register a new buffer to get logged into tombstones for crashes.
 *
 * It will be added to both the tombstone proto in the crash_detail field, and
 * in the tombstone text format.
 *
 * Tombstone proto definition:
 *   https://cs.android.com/android/platform/superproject/main/+/main:system/core/debuggerd/proto/tombstone.proto
 *
 * An app can get hold of these for any `REASON_CRASH_NATIVE` instance of
 * `android.app.ApplicationExitInfo`.
 *
 * https://developer.android.com/reference/android/app/ApplicationExitInfo#getTraceInputStream()

 * The lifetime of name and data has to be valid until the program crashes, or until
 * android_crash_detail_unregister is called.
 *
 * Example usage:
 *   const char* stageName = "garbage_collection";
 *   crash_detail_t* cd = android_crash_detail_register("stage", stageName, strlen(stageName));
 *   do_garbage_collection();
 *   android_crash_detail_unregister(cd);
 *
 * If this example crashes in do_garbage_collection, a line will show up in the textual representation of the tombstone:
 *   Extra crash detail: stage: 'garbage_collection'
 *
 * Introduced in API 35.
 *
 * \param name identifying name for this extra data.
 *             this should generally be a human-readable UTF-8 string, but we are treating
 *             it as arbitrary bytes because it could be corrupted by the crash.
 * \param name_size number of bytes of the buffer pointed to by name
 * \param data a buffer containing the extra detail bytes, if null the crash detail
 *             is disabled until android_crash_detail_replace_data replaces it with
 *             a non-null pointer.
 * \param data_size number of bytes of the buffer pointed to by data
 *
 * \return a handle to the extra crash detail.
 */

#if __BIONIC_AVAILABILITY_GUARD(35)
crash_detail_t* _Nullable android_crash_detail_register(
    const void* _Nonnull name, size_t name_size, const void* _Nullable data, size_t data_size) __INTRODUCED_IN(35);

/**
 * Unregister crash detail from being logged into tombstones.
 *
 * After this function returns, the lifetime of the objects crash_detail was
 * constructed from no longer needs to be valid.
 *
 * Introduced in API 35.
 *
 * \param crash_detail the crash_detail that should be removed.
 */
void android_crash_detail_unregister(crash_detail_t* _Nonnull crash_detail) __INTRODUCED_IN(35);

/**
 * Replace data of crash detail.
 *
 * This is more efficient than using android_crash_detail_unregister followed by
 * android_crash_detail_register. If you very frequently need to swap out the data,
 * you can hold onto the crash_detail.
 *
 * Introduced in API 35.
 *
 * \param data the new buffer containing the extra detail bytes, or null to disable until
 *             android_crash_detail_replace_data is called again with non-null data.
 * \param data_size the number of bytes of the buffer pointed to by data.
 */
void android_crash_detail_replace_data(crash_detail_t* _Nonnull crash_detail, const void* _Nullable data, size_t data_size) __INTRODUCED_IN(35);

/**
 * Replace name of crash detail.
 *
 * This is more efficient than using android_crash_detail_unregister followed by
 * android_crash_detail_register. If you very frequently need to swap out the name,
 * you can hold onto the crash_detail.
 *
 * Introduced in API 35.
 *
 * \param name identifying name for this extra data.
 * \param name_size number of bytes of the buffer pointed to by name
 */
void android_crash_detail_replace_name(crash_detail_t* _Nonnull crash_detail, const void* _Nonnull name, size_t name_size) __INTRODUCED_IN(35);
#endif /* __BIONIC_AVAILABILITY_GUARD(35) */


__END_DECLS
