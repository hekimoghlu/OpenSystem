/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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

#ifndef TEST_INTEROP_CXX_VALUE_WITNESS_TABLE_INPUTS_COPY_CONSTRUCTORS_H
#define TEST_INTEROP_CXX_VALUE_WITNESS_TABLE_INPUTS_COPY_CONSTRUCTORS_H

struct __attribute__((language_attr("import_unsafe")))
HasUserProvidedCopyConstructor {
  int numCopies;
  HasUserProvidedCopyConstructor(int numCopies = 0) : numCopies(numCopies) {}
  HasUserProvidedCopyConstructor(const HasUserProvidedCopyConstructor &other)
      : numCopies(other.numCopies + 1) {}
};

struct HasNonTrivialImplicitCopyConstructor {
  HasUserProvidedCopyConstructor box;
  HasNonTrivialImplicitCopyConstructor()
      : box(HasUserProvidedCopyConstructor()) {}
};

struct HasNonTrivialDefaultCopyConstructor {
  HasUserProvidedCopyConstructor box;
  HasNonTrivialDefaultCopyConstructor()
      : box(HasUserProvidedCopyConstructor()) {}
  HasNonTrivialDefaultCopyConstructor(
      const HasNonTrivialDefaultCopyConstructor &) = default;
};

struct HasCopyConstructorWithDefaultArgs {
  int value;
  HasCopyConstructorWithDefaultArgs(int value) : value(value) {}

  HasCopyConstructorWithDefaultArgs(
      const HasCopyConstructorWithDefaultArgs &other, int value = 1)
      : value(other.value + value) {}

  HasCopyConstructorWithDefaultArgs(HasCopyConstructorWithDefaultArgs &&) =
      default;
};

struct HasCopyConstructorWithOneParameterWithDefaultArg {
  int numCopies;

  HasCopyConstructorWithOneParameterWithDefaultArg(int numCopies)
      : numCopies(numCopies) {}

  HasCopyConstructorWithOneParameterWithDefaultArg(
      const HasCopyConstructorWithOneParameterWithDefaultArg &other =
          HasCopyConstructorWithOneParameterWithDefaultArg{1})
      : numCopies(other.numCopies + 1) {}
};

// Make sure that we don't crash on struct templates with copy-constructors.
template <typename T> struct S {
  S(S const &) {}
};

#endif // TEST_INTEROP_CXX_VALUE_WITNESS_TABLE_INPUTS_COPY_CONSTRUCTORS_H
