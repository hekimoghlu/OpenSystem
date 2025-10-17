/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 13, 2024.
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

#ifndef TEST_INTEROP_CXX_CLASS_INLINE_FUNCTION_THROUGH_MEMBER_INPUTS_METHOD_CALLS_FUNCTION_H
#define TEST_INTEROP_CXX_CLASS_INLINE_FUNCTION_THROUGH_MEMBER_INPUTS_METHOD_CALLS_FUNCTION_H

inline int increment(int t) { return t + 1; }

struct Incrementor {
  int callIncrement(int value) { return increment(value); }
};

inline int callMethod(int value) { return Incrementor().callIncrement(value); }

class __attribute__((language_attr("import_reference")))
__attribute__((language_attr("retain:immortal")))
__attribute__((language_attr("release:immortal")))
__attribute__((language_attr("unsafe"))) Cell {
public:
  bool is_marked() const { return m_marked; }
  void set_marked(bool b) { m_marked = b; }

private:
  bool m_marked : 1 {false};
};

inline Cell *_Nonnull createCell() { return new Cell{}; }

#endif // TEST_INTEROP_CXX_CLASS_INLINE_FUNCTION_THROUGH_MEMBER_INPUTS_METHOD_CALLS_FUNCTION_H
