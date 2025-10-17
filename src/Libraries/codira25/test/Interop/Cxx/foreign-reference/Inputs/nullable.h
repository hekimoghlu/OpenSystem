/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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

#ifndef TEST_INTEROP_CXX_FOREIGN_REFERENCE_INPUTS_NULLABLE_H
#define TEST_INTEROP_CXX_FOREIGN_REFERENCE_INPUTS_NULLABLE_H

#include <stdlib.h>
#include <new>

struct __attribute__((language_attr("import_reference")))
__attribute__((language_attr("retain:immortal")))
__attribute__((language_attr("release:immortal"))) Empty {
  int test() const { return 42; }

  static Empty *create() { return new (malloc(sizeof(Empty))) Empty(); }
};

void mutateIt(Empty &) {}

struct __attribute__((language_attr("import_reference")))
__attribute__((language_attr("retain:immortal")))
__attribute__((language_attr("release:immortal"))) IntPair {
  int a = 1;
  int b = 2;

  int test() const { return b - a; }

  static IntPair *create() { return new (malloc(sizeof(IntPair))) IntPair(); }
};

void mutateIt(IntPair *x) {
  x->a = 2;
  x->b = 4;
}

#endif // TEST_INTEROP_CXX_FOREIGN_REFERENCE_INPUTS_NULLABLE_H
