/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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
#ifndef mbmalloc_h
#define mbmalloc_h

#include <stddef.h>

// This file defines a default implementation of the mbmalloc API, using system
// malloc. To test with another malloc, supply an override .dylib that exports
// these symbols.

extern "C" {

void* mbmalloc(size_t);
void* mbmemalign(size_t, size_t);
void mbfree(void*, size_t);
void* mbrealloc(void*, size_t, size_t);
void mbscavenge();
    
}

// Catch accidental benchmark allocation through malloc and free. All benchmark
// code should use mbmalloc / mbfree, to call the instrumented malloc we're
// benchmarking against.

#define malloc error
#define free error
#define realloc error

#endif // mbmalloc_h
