/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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

#ifndef TEST_INTEROP_CXX_FOREIGN_REFERENCE_INPUTS_WITNESS_TABLE_H
#define TEST_INTEROP_CXX_FOREIGN_REFERENCE_INPUTS_WITNESS_TABLE_H

#include <stdlib.h>
#if defined(_WIN32)
inline void *operator new(size_t, void *p) { return p; }
#else
#include <new>
#endif

struct __attribute__((language_attr("import_reference")))
__attribute__((language_attr("retain:immortal")))
__attribute__((language_attr("release:immortal"))) CxxLinkedList {
  int value = 3;

  CxxLinkedList * _Nullable next() {
    if (value == 3)
      return nullptr;

    return this + 1;
  }
};

CxxLinkedList * _Nonnull makeLinkedList() {
  CxxLinkedList *buff = (CxxLinkedList *)malloc(sizeof(CxxLinkedList) * 4);
  buff[0].value = 0;
  buff[1].value = 1;
  buff[2].value = 2;
  buff[3].value = 3;
  return buff;
}

struct __attribute__((language_attr("import_reference")))
__attribute__((language_attr("retain:immortal")))
__attribute__((language_attr("release:immortal"))) MyCxxSequence {
  CxxLinkedList * _Nullable list = nullptr;

  CxxLinkedList * _Nullable next() {
    if (list->value == 3)
      return nullptr;

    auto * _Nullable tmp = list;
    list = tmp + 1;
    return tmp;
  }
};

MyCxxSequence * _Nonnull makeSequence() {
  CxxLinkedList *buff = (CxxLinkedList *)malloc(sizeof(CxxLinkedList) * 4);
  buff[0].value = 0;
  buff[1].value = 1;
  buff[2].value = 2;
  buff[3].value = 3;

  MyCxxSequence *seq = (MyCxxSequence *)malloc(sizeof(MyCxxSequence));
  seq->list = buff;
  return seq;
}

#endif // TEST_INTEROP_CXX_FOREIGN_REFERENCE_INPUTS_WITNESS_TABLE_H
