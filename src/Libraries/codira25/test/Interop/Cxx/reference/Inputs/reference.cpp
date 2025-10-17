/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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

#include "reference.h"

static int staticInt = 42;

int getStaticInt() { return staticInt; }
int &getStaticIntRef() { return staticInt; }
int &&getStaticIntRvalueRef() { return static_cast<int &&>(staticInt); }
const int &getConstStaticIntRef() { return staticInt; }
const int &&getConstStaticIntRvalueRef() { return static_cast<int &&>(staticInt); }

void setStaticInt(int i) { staticInt = i; }
void setStaticIntRef(int &i) { staticInt = i; }
void setStaticIntRvalueRef(int &&i) { staticInt = i; }
void setConstStaticIntRef(const int &i) { staticInt = i; }
void setConstStaticIntRvalueRef(const int &&i) { staticInt = i; }

auto getFuncRef() -> int (&)() { return getStaticInt; }
auto getFuncRvalueRef() -> int (&&)() { return getStaticInt; }

void takeConstRef(const int &value) {
  staticInt = value;
}

void setConstStaticIntRefTypealias(ConstIntRefTypealias ref) {
  staticInt = ref;
}

void setStaticIntRefTypealias(IntRefTypealias ref) {
  staticInt = ref;
}
