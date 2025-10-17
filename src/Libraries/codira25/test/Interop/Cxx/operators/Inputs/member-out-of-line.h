/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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

#ifndef TEST_INTEROP_CXX_OPERATORS_INPUTS_MEMBER_OUT_OF_LINE_H
#define TEST_INTEROP_CXX_OPERATORS_INPUTS_MEMBER_OUT_OF_LINE_H

struct LoadableIntWrapper {
  int value;
  LoadableIntWrapper operator+(LoadableIntWrapper rhs) const;
  int operator()() const;
  int operator()(int x) const;
  int operator()(int x, int y) const;
};

struct __attribute__((language_attr("import_owned"))) AddressOnlyIntWrapper {
  int value;

  AddressOnlyIntWrapper(int value) : value(value) {}
  AddressOnlyIntWrapper(const AddressOnlyIntWrapper &other) : value(other.value) {}

  int operator()() const;
  int operator()(int x) const;
  int operator()(int x, int y) const;
};

struct ReadWriteIntArray {
private:
  int values[5] = { 1, 2, 3, 4, 5 };

public:
  const int &operator[](int x) const;
  int &operator[](int x);
};

struct __attribute__((language_attr("import_owned"))) NonTrivialIntArrayByVal {
  NonTrivialIntArrayByVal(int first) { values[0] = first; }
  NonTrivialIntArrayByVal(const NonTrivialIntArrayByVal &other) {
    for (int i = 0; i < 5; i++)
      values[i] = other.values[i];
  }
  int operator[](int x);

  // For testing purposes.
  void setValueAtIndex(int value, unsigned i) { values[i] = value; }

private:
  int values[5] = { 1, 2, 3, 4, 5 };
};

struct ClassWithOperatorEqualsParamUnnamed {
  bool operator==(const ClassWithOperatorEqualsParamUnnamed &) const;
  bool operator!=(const ClassWithOperatorEqualsParamUnnamed &) const;
};

#endif
