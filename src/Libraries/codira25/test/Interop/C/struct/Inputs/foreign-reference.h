/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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

#ifndef TEST_INTEROP_C_INPUTS_FOREIGN_REFERENCE_H
#define TEST_INTEROP_C_INPUTS_FOREIGN_REFERENCE_H

#include <stdlib.h>

#if __has_feature(nullability)
// Provide macros to temporarily suppress warning about the use of
// _Nullable and _Nonnull.
# define LANGUAGE_BEGIN_NULLABILITY_ANNOTATIONS                                   \
  _Pragma("clang diagnostic push")                                             \
  _Pragma("clang diagnostic ignored \"-Wnullability-extension\"")              \
  _Pragma("clang assume_nonnull begin")
# define LANGUAGE_END_NULLABILITY_ANNOTATIONS                                     \
  _Pragma("clang diagnostic pop")                                              \
  _Pragma("clang assume_nonnull end")

#else
// #define _Nullable and _Nonnull to nothing if we're not being built
// with a compiler that supports them.
# define _Nullable
# define _Nonnull
# define _Null_unspecified
# define LANGUAGE_BEGIN_NULLABILITY_ANNOTATIONS
# define LANGUAGE_END_NULLABILITY_ANNOTATIONS
#endif

LANGUAGE_BEGIN_NULLABILITY_ANNOTATIONS

struct
    __attribute__((language_attr("import_as_ref")))
    __attribute__((language_attr("retain:LCRetain")))
    __attribute__((language_attr("release:LCRelease")))
LocalCount {
  int value;
};

static inline struct LocalCount *createLocalCount() {
  struct LocalCount *ptr = malloc(sizeof(struct LocalCount));
  ptr->value = 1;
  return ptr;
}

static inline void LCRetain(struct LocalCount *x) { x->value++; }
static inline void LCRelease(struct LocalCount *x) { x->value--; }

static int globalCount = 0;

struct
    __attribute__((language_attr("import_as_ref")))
    __attribute__((language_attr("retain:GCRetain")))
    __attribute__((language_attr("release:GCRelease")))
GlobalCount {};

static inline struct GlobalCount *createGlobalCount() {
  globalCount++;
  return malloc(sizeof(struct GlobalCount));
}

static inline void GCRetain(struct GlobalCount *x) { globalCount++; }
static inline void GCRelease(struct GlobalCount *x) { globalCount--; }

LANGUAGE_END_NULLABILITY_ANNOTATIONS

#endif // TEST_INTEROP_C_INPUTS_FOREIGN_REFERENCE_H
