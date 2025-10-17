/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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

#ifndef TEST_INTEROP_CXX_CLASS_INPUTS_MEMBERWISE_INITIALIZER_H
#define TEST_INTEROP_CXX_CLASS_INPUTS_MEMBERWISE_INITIALIZER_H

template <typename T>
struct TemplatedType {};


struct StructPrivateOnly {
private:
  int varPrivate;
};

struct StructPublicOnly {
  int varPublic;
};

struct StructEmptyPrivateSection {
  int varPublic;
private:
};

struct StructPublicAndPrivate {
  int varPublic;
private:
  int varPrivate;
};

struct StructWithUnimportedMemberFunction {
  int varPublic;
  int StructWithUnimportedMemberFunction::* unimportedMemberFunction();
};

class ClassPrivateOnly {
  int varPrivate;
};

class ClassPublicOnly {
public:
  int varPublic;
};

class ClassEmptyPublicSection {
  int varPrivate;
public:
};

class ClassPrivateAndPublic {
  int varPrivate;
public:
  int varPublic;
};

struct ClassWithUnimportedMemberFunction {
public:
  int varPublic;
  int ClassWithUnimportedMemberFunction::* unimportedMemberFunction();
};

struct ClassWithTemplatedFunction {
public:
  int varPublic;

  template <int I>
  void foo();
};

struct ClassWithTemplatedUsingDecl {
public:
  int varPublic;

  template <typename T>
  using MyUsing = TemplatedType<T>;
};

#endif
