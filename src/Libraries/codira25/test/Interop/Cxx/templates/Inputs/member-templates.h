/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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

#ifndef TEST_INTEROP_CXX_TEMPLATES_INPUTS_MEMBER_TEMPLATES_H
#define TEST_INTEROP_CXX_TEMPLATES_INPUTS_MEMBER_TEMPLATES_H

struct HasMemberTemplates {
  template <class T> T addSameTypeParams(T a, T b) { return a + b; }

  template <class T, class U> T addMixedTypeParams(T a, U b) { return a + b; }

  template <class T, class U> int addAll(int a, T b, U c) { return a + b + c; }

  template <class T> T passThrough(T val) { return val; }

  template <class T> T passThroughConst(const T val) { return val; }

  template <class T> T passThroughOnConst(T val) const { return val; }

  template <class T> T passThroughConstOnConst(const T val) const {
    return val;
  }

  template <class T> void doNothingConstRef(const T &val) {}

  template <class T> void make42Ref(T &val) {}
};

template <class T> struct TemplateClassWithMemberTemplates {
  T value;

  template <class U> void setValue(U val) { value = val; }

  template<class U> TemplateClassWithMemberTemplates<U> toOtherSpec(const U& u) const {
    return {u};
  }

  TemplateClassWithMemberTemplates(T val) : value(val) {}
};

using IntWrapper = TemplateClassWithMemberTemplates<int>;

struct HasStaticMemberTemplates {
  template <class T> static T add(T a, T b) { return a + b; }
  template <class T, class U> static T addTwoTemplates(T a, U b) { return a + b; }
  template <class T> static T removeReference(T &a) { return a; }
};

template <typename T>
struct MyTemplatedStruct {};

struct HasTemplatedField {
  MyTemplatedStruct<int> x;
};

struct HasNestedInstantiation {
  template <typename T>
  struct MyNestedTemplatedStruct {};

  using NestedInst = MyTemplatedStruct<MyNestedTemplatedStruct<int>>;
};

namespace NS {
struct HasNestedInstantiation {
  template <typename T>
  struct MyNestedTemplatedStruct {};

  using NestedInst = MyTemplatedStruct<MyNestedTemplatedStruct<int>>;
};
}

template <typename A, typename R = TemplateClassWithMemberTemplates<A>>
struct HasUninstantiatableTemplateMember {
  R *pointer; // R cannot be instantiated here, because R is an incomplete type,
              // so this should be imported as OpaquePointer.
};

struct HasTemplateInstantiationWithForwardDecl {
  class NoDefinition;

  HasUninstantiatableTemplateMember<NoDefinition> noDefMember;
};

#endif // TEST_INTEROP_CXX_TEMPLATES_INPUTS_MEMBER_TEMPLATES_H
