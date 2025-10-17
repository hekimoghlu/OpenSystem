/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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

#include <pthread.h>
#include <stdatomic.h>

#include <private/bionic_globals.h>
#include <private/bionic_malloc_dispatch.h>

// Function prototypes.
bool InitSharedLibrary(void* impl_handle, const char* shared_lib, const char* prefix,
                       MallocDispatch* dispatch_table);

void* LoadSharedLibrary(const char* shared_lib, const char* prefix, MallocDispatch* dispatch_table);

bool FinishInstallHooks(libc_globals* globals, const char* options, const char* prefix);

// Lock for globals, to guarantee that only one thread is doing a mutate.
extern pthread_mutex_t gGlobalsMutateLock;
extern _Atomic bool gGlobalsMutating;

// Function hooks instantiations, used by dispatch-table allocators to install themselves.
void SetGlobalFunctions(void* functions[]);
