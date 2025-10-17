/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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

//===--- TracingSignpost.cpp - Tracing with the signpost API -------*- C++ -*-//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//
//
// Concurrency tracing implemented with the os_signpost API.
//
//===----------------------------------------------------------------------===//

#if LANGUAGE_STDLIB_CONCURRENCY_TRACING

#include "TracingSignpost.h"
#include <stdio.h>

#define LANGUAGE_LOG_CONCURRENCY_SUBSYSTEM "com.apple.code.concurrency"
#define LANGUAGE_LOG_ACTOR_CATEGORY "Actor"
#define LANGUAGE_LOG_TASK_CATEGORY "Task"

namespace language {
namespace concurrency {
namespace trace {

os_log_t ActorLog;
os_log_t TaskLog;
language::once_t LogsToken;
bool TracingEnabled;

void setupLogs(void *unused) {
  if (!language::runtime::trace::shouldEnableTracing()) {
    TracingEnabled = false;
    return;
  }

  TracingEnabled = true;
  ActorLog = os_log_create(LANGUAGE_LOG_CONCURRENCY_SUBSYSTEM,
                           LANGUAGE_LOG_ACTOR_CATEGORY);
  TaskLog = os_log_create(LANGUAGE_LOG_CONCURRENCY_SUBSYSTEM,
                          LANGUAGE_LOG_TASK_CATEGORY);
}

} // namespace trace
} // namespace concurrency
} // namespace language

#endif
