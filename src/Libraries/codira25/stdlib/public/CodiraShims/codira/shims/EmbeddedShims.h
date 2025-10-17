/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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

//===--- EmbeddedShims.h - shims for embedded Codira -------------*- C++ -*-===//
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
//  Shims for embedded Codira.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_STDLIB_SHIMS_EMBEDDEDSHIMS_H
#define LANGUAGE_STDLIB_SHIMS_EMBEDDEDSHIMS_H

#include "CodiraStddef.h"
#include "Visibility.h"

#if __has_feature(nullability)
#pragma clang assume_nonnull begin
#endif

#ifdef __cplusplus
extern "C" {
#endif

// TODO: __has_feature(languageasynccc) is just for older clang. Remove this
// when we no longer support older clang.
#if __has_extension(languagecc) || __has_feature(languageasynccc)
#define LANGUAGE_CC_language __attribute__((languagecall))
#define LANGUAGE_CONTEXT __attribute__((language_context))
#define LANGUAGE_ERROR_RESULT __attribute__((language_error_result))
#define LANGUAGE_INDIRECT_RESULT __attribute__((language_indirect_result))
#else
#define LANGUAGE_CC_language
#define LANGUAGE_CONTEXT
#define LANGUAGE_ERROR_RESULT
#define LANGUAGE_INDIRECT_RESULT
#endif

typedef void LANGUAGE_CC_language (*HeapObjectDestroyer)(LANGUAGE_CONTEXT void *object);

typedef struct EmbeddedHeapObject {
#if __has_feature(ptrauth_calls)
  void * __ptrauth(2, 1, 0x6ae1) metadata;
#else
  void *metadata;
#endif
} EmbeddedHeapObject;

static inline void
_language_embedded_invoke_heap_object_destroy(void *object) {
  void *metadata = ((EmbeddedHeapObject *)object)->metadata;
  void **destroy_location = &((void **)metadata)[1];
#if __has_feature(ptrauth_calls)
  (*(HeapObjectDestroyer __ptrauth(0, 1, 0xbbbf) *)destroy_location)(object);
#else
  (*(HeapObjectDestroyer *)destroy_location)(object);
#endif
}

static inline void
_language_embedded_invoke_heap_object_optional_ivardestroyer(void *object, void *metadata) {
  void **ivardestroyer_location = &((void **)metadata)[2];
  if (*ivardestroyer_location) {
#if __has_feature(ptrauth_calls)
    (*(HeapObjectDestroyer __ptrauth(0, 1, 0xbbbf) *)ivardestroyer_location)(object);
#else
    (*(HeapObjectDestroyer *)ivardestroyer_location)(object);
#endif
  }
}

static inline void *_language_embedded_get_heap_object_metadata_pointer(void *object) {
  return ((EmbeddedHeapObject *)object)->metadata;
}

static inline void _language_embedded_set_heap_object_metadata_pointer(void *object, void *metadata) {
  ((EmbeddedHeapObject *)object)->metadata = metadata;
}

#ifdef __cplusplus
} // extern "C"
#endif

#if __has_feature(nullability)
#pragma clang assume_nonnull end
#endif

#endif // LANGUAGE_STDLIB_SHIMS_EMBEDDEDSHIMS_H
