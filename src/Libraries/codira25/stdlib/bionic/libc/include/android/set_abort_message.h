/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 18, 2024.
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
 * @file android/set_abort_message.h
 * @brief The android_set_abort_message() function.
 */

#include <sys/cdefs.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

__BEGIN_DECLS

typedef struct crash_detail_t crash_detail_t;

/**
 * android_set_abort_message() sets the abort message passed to
 * [debuggerd](https://source.android.com/devices/tech/debug/native-crash)
 * for inclusion in any crash.
 *
 * This is meant for use by libraries that deliberately abort so that they can
 * provide an explanation. It is used within bionic to implement assert() and
 * all FORTIFY and fdsan failures.
 *
 * The message appears directly in logcat at the time of crash. It will
 * also be added to both the tombstone proto in the crash_detail field, and
 * in the tombstone text format.
 *
 * Tombstone proto definition:
 *   https://cs.android.com/android/platform/superproject/main/+/main:system/core/debuggerd/proto/tombstone.proto
 *
 * An app can get hold of these for any `REASON_CRASH_NATIVE` instance of
 * `android.app.ApplicationExitInfo`.
 *  https://developer.android.com/reference/android/app/ApplicationExitInfo#getTraceInputStream()
 *
 * The given message is copied at the time this function is called, and does
 * not need to be valid until the crash actually happens, but typically this
 * function is called immediately before aborting. See <android/crash_detail.h>
 * for API more suited to the use case where the caller doesn't _expect_ a
 * crash but would like to see the information _if_ a crash happens.
 */
void android_set_abort_message(const char* _Nullable __msg);

__END_DECLS
