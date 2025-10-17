/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 30, 2023.
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

#ifndef TEST_INTEROP_CXX_CLASS_INPUTS_MEMORY_LAYOUT_H
#define TEST_INTEROP_CXX_CLASS_INPUTS_MEMORY_LAYOUT_H

#include <cstddef>
#include <cstdint>

class PrivateMemberLayout {
  uint32_t a;

public:
  uint32_t b;
};

inline size_t sizeOfPrivateMemberLayout() {
  return sizeof(PrivateMemberLayout);
}

inline size_t offsetOfPrivateMemberLayout_b() {
  return offsetof(PrivateMemberLayout, b);
}

#endif
