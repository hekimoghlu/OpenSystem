/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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

#ifndef TEST_INTEROP_CXX_FOREIGN_REFERENCE_INPUTS_REFERENCE_COUNTED_H
#define TEST_INTEROP_CXX_FOREIGN_REFERENCE_INPUTS_REFERENCE_COUNTED_H

#include <stdlib.h>
#include <new>

#include "visibility.h"

LANGUAGE_BEGIN_NULLABILITY_ANNOTATIONS

static int finalLocalRefCount = 100;

namespace NS {

struct __attribute__((language_attr("import_as_ref")))
__attribute__((language_attr("retain:LCRetain")))
__attribute__((language_attr("release:LCRelease"))) LocalCount final {
  int value = 0;

  static LocalCount *create() {
    return new (malloc(sizeof(LocalCount))) LocalCount();
  }

  int returns42() { return 42; }
  int constMethod() const { return 42; }
};

}

inline void LCRetain(NS::LocalCount *x) {
  x->value++;
  finalLocalRefCount = x->value;
}
inline void LCRelease(NS::LocalCount *x) {
  x->value--;
  finalLocalRefCount = x->value;
}

static int globalCount = 0;

struct __attribute__((language_attr("import_as_ref")))
__attribute__((language_attr("retain:GCRetain")))
__attribute__((language_attr("release:GCRelease"))) GlobalCount {
  static GlobalCount *create() {
    return new (malloc(sizeof(GlobalCount))) GlobalCount();
  }
};

inline void GCRetain(GlobalCount *x) { globalCount++; }
inline void GCRelease(GlobalCount *x) { globalCount--; }

struct __attribute__((language_attr("import_as_ref")))
__attribute__((language_attr("retain:GCRetainNullableInit")))
__attribute__((language_attr("release:GCReleaseNullableInit")))
GlobalCountNullableInit {
  static GlobalCountNullableInit *_Nullable create(bool wantNullptr) {
    if (wantNullptr)
      return nullptr;
    return new (malloc(sizeof(GlobalCountNullableInit)))
        GlobalCountNullableInit();
  }
};

inline void GCRetainNullableInit(GlobalCountNullableInit *x) { globalCount++; }
inline void GCReleaseNullableInit(GlobalCountNullableInit *x) { globalCount--; }

LANGUAGE_END_NULLABILITY_ANNOTATIONS

#endif // TEST_INTEROP_CXX_FOREIGN_REFERENCE_INPUTS_REFERENCE_COUNTED_H
