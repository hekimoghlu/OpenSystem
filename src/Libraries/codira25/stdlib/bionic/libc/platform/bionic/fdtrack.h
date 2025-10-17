/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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

#include <sys/cdefs.h>
#include <stdbool.h>
#include <stdint.h>

__BEGIN_DECLS

// Types of an android_fdtrack_event.
enum android_fdtrack_event_type {
  // File descriptor creation: create is the active member of android_fdtrack_event::data.
  ANDROID_FDTRACK_EVENT_TYPE_CREATE,

  // File descriptor closed.
  ANDROID_FDTRACK_EVENT_TYPE_CLOSE,
};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
struct android_fdtrack_event {
  // File descriptor for which this event occurred.
  int fd;

  // Type of event: this is one of the enumerators of android_fdtrack_event_type.
  uint8_t type;

  // Data for the event.
  union {
    struct {
      const char* function_name;
    } create;
  } data;
};
#pragma clang diagnostic pop

// Callback invoked upon file descriptor creation/closure.
typedef void (*_Nullable android_fdtrack_hook_t)(struct android_fdtrack_event* _Nullable);

// Register a hook which is called to track fd lifecycle events.
// Set value to null to disable tracking.
bool android_fdtrack_compare_exchange_hook(android_fdtrack_hook_t* _Nonnull expected,
                                           android_fdtrack_hook_t value) __INTRODUCED_IN(30);

// Enable/disable fdtrack *on the current thread*.
// This is primarily useful when performing operations which you don't want to track
// (e.g. when emitting already-recorded information).
bool android_fdtrack_get_enabled() __INTRODUCED_IN(30);
bool android_fdtrack_set_enabled(bool new_value) __INTRODUCED_IN(30);

// Globally enable/disable fdtrack.
// This is primaryily useful to reenable fdtrack after it's been automatically disabled post-fork.
void android_fdtrack_set_globally_enabled(bool new_value) __INTRODUCED_IN(31);

__END_DECLS
