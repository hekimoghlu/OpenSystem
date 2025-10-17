/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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

#ifndef TEST_INTEROP_CXX_CLASS_CUSTOM_NEW_OPERATOR_H
#define TEST_INTEROP_CXX_CLASS_CUSTOM_NEW_OPERATOR_H

#include <cstddef>
#include <cstdint>

struct container_new_t {};

inline void *operator new(size_t, void *p, container_new_t) { return p; }

struct MakeMe {
  int x;
};

inline MakeMe *callsCustomNew() {
  char buffer[8];
  return new (buffer, container_new_t()) MakeMe;
}

#endif // TEST_INTEROP_CXX_CLASS_CUSTOM_NEW_OPERATOR_H
