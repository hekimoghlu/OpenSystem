/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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

#pragma once

struct One {
  int method(void) const { return 1; }
  int operator[](int i) const { return 1; }
};

struct IOne : One {
  int methodI(void) const { return -1; }
};

struct IIOne : IOne {
  int methodII(void) const { return -11; }
};

struct IIIOne : IIOne {
  int methodIII(void) const { return -111; }
};

class Base {
public:
  bool baseMethod() const { return true; }
};

namespace Bar {
class Derived : public Base {};
} // namespace Bar
