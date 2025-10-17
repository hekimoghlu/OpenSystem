/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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

// Enables LLINT tracing.
// - Prints every instruction executed if Options::traceLLIntExecution() is enabled.
// - Prints some information for some of the more subtle slow paths if
//   Options::traceLLIntSlowPath() is enabled.
#define LLINT_TRACING 0

// Disable inline allocation in the interpreter. This is great if you're changing
// how the GC allocates.
#if ENABLE(ALLOCATION_LOGGING)
#define LLINT_ALWAYS_ALLOCATE_SLOW 1
#else
#define LLINT_ALWAYS_ALLOCATE_SLOW 0
#endif
