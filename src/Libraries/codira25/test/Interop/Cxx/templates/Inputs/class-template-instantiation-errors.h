/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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

#ifndef TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_INSTANTIATION_ERRORS_H
#define TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_INSTANTIATION_ERRORS_H

template<class T>
struct MagicWrapper {
  T t;
  int getValuePlusArg(int arg) const { return t.getValue() + arg; }
};

template<class T>
struct MagicWrapperWithExplicitCtor {
  T t;
  MagicWrapperWithExplicitCtor(T t) : t(t) {}
};

struct IntWrapper {
  int value;
  int getValue() const { return value; }
};

template<class T>
struct CannotBeInstantianted {
  T value;

  CannotBeInstantianted(char, T value) { value.doesNotExist(); }
  CannotBeInstantianted(char, char) { memberWrongType(); }
  CannotBeInstantianted(T value) : value(value) {}

  void callsMethodWithError() { memberWrongType(); }

  void memberWrongType() { value.doesNotExist(); }

  void argWrongType(T t) { t.doesNotExist(); }

  int getOne() { return 1; }
  int incValue() { return value.value + getOne(); }
  int incValue(T t) { return t.value + getOne(); }
};

#endif // TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_INSTANTIATION_ERRORS_H
