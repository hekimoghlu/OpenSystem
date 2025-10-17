/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 6, 2025.
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

#ifndef TEST_INTEROP_CXX_IMPLEMENTATION_ONLY_IMPORTS_INPUTS_HELPER_H
#define TEST_INTEROP_CXX_IMPLEMENTATION_ONLY_IMPORTS_INPUTS_HELPER_H

inline int getFortyTwo() { return 42; }

class MagicWrapper {
public:
  int _number;
  MagicWrapper(){_number = 2;};
  MagicWrapper(int number) : _number(number){};
  MagicWrapper operator - (MagicWrapper other) {
      return MagicWrapper{_number - other._number};
  }

  int baseMethod() const { return 42; }
};

inline MagicWrapper operator + (MagicWrapper lhs, MagicWrapper rhs) {
  return MagicWrapper{lhs._number + rhs._number};
}

class MagicWrapperDerived: public MagicWrapper {
public:
  MagicWrapperDerived() { };
};

#endif // TEST_INTEROP_CXX_IMPLEMENTATION_ONLY_IMPORTS_INPUTS_HELPER_H
