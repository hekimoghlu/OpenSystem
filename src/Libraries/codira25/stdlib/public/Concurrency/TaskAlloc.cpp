/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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

//===--- TaskAlloc.cpp - Task-local stack allocator -----------------------===//
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
// A task-local allocator that obeys a stack discipline.
//
// Because allocation is task-local, and there's at most one thread
// running a task at once, no synchronization is required.
//
//===----------------------------------------------------------------------===//

#include "TaskPrivate.h"
#include "language/ABI/Task.h"
#include "language/Runtime/Concurrency.h"

#include <stdlib.h>

using namespace language;

namespace {

struct GlobalAllocator {
  TaskAllocator allocator;
  void *spaceForFirstSlab[64];

  GlobalAllocator() : allocator(spaceForFirstSlab, sizeof(spaceForFirstSlab)) {}
};

} // end anonymous namespace

static TaskAllocator &allocator(AsyncTask *task) {
  if (task)
    return task->Private.get().Allocator;

#if !LANGUAGE_CONCURRENCY_EMBEDDED
  // FIXME: this fall-back shouldn't be necessary, but it's useful
  // for now, since the current execution tests aren't setting up a task
  // properly.
  static GlobalAllocator global;
  return global.allocator;
#else
  puts("global allocator fallback not available\n");
  abort();
#endif
}

void *language::language_task_alloc(size_t size) {
  return allocator(language_task_getCurrent()).alloc(size);
}

void *language::_language_task_alloc_specific(AsyncTask *task, size_t size) {
  return allocator(task).alloc(size);
}

void language::language_task_dealloc(void *ptr) {
  allocator(language_task_getCurrent()).dealloc(ptr);
}

void language::_language_task_dealloc_specific(AsyncTask *task, void *ptr) {
  allocator(task).dealloc(ptr);
}

void language::language_task_dealloc_through(void *ptr) {
  allocator(language_task_getCurrent()).deallocThrough(ptr);
}
void *language::language_job_allocate(Job *job, size_t size) {
  if (!job->isAsyncTask())
    return nullptr;

  return allocator(static_cast<AsyncTask *>(job)).alloc(size);
}

void language::language_job_deallocate(Job *job, void *ptr) {
  if (!job->isAsyncTask())
    return;

  allocator(static_cast<AsyncTask *>(job)).dealloc(ptr);
}
