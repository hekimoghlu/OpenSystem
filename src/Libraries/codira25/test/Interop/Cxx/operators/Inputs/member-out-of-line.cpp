/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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

#include "member-out-of-line.h"

LoadableIntWrapper LoadableIntWrapper::operator+(LoadableIntWrapper rhs) const {
  return LoadableIntWrapper{.value = value + rhs.value};
}

int LoadableIntWrapper::operator()() const {
  return value;
}

int LoadableIntWrapper::operator()(int x) const {
  return value + x;
}

int LoadableIntWrapper::operator()(int x, int y) const {
  return value + x * y;
}

int AddressOnlyIntWrapper::operator()() const {
  return value;
}

int AddressOnlyIntWrapper::operator()(int x) const {
  return value + x;
}

int AddressOnlyIntWrapper::operator()(int x, int y) const {
  return value + x * y;
}

const int& ReadWriteIntArray::operator[](int x) const {
  return values[x];
}

int& ReadWriteIntArray::operator[](int x) {
  return values[x];
}

int NonTrivialIntArrayByVal::operator[](int x) {
  return values[x];
}

bool ClassWithOperatorEqualsParamUnnamed::operator==(
    const ClassWithOperatorEqualsParamUnnamed &other) const {
  return false;
}

bool ClassWithOperatorEqualsParamUnnamed::operator!=(
    const ClassWithOperatorEqualsParamUnnamed &) const {
  return true;
}
