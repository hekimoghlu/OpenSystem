/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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

#ifndef TEST_INTEROP_CXX_CLASS_AMBIGUOUS_METHOD_METHODS_H
#define TEST_INTEROP_CXX_CLASS_AMBIGUOUS_METHOD_METHODS_H

struct HasAmbiguousMethods {

  // One input (const first)
  int increment(int a) const {
    return a + 1;
  }

  int increment(int a) {
    ++mutableMethodsCalledCount;
    return a + 1;
  }

  // Multiple input with out param
  void increment(int a, int b, int &c) {
    ++mutableMethodsCalledCount;
    c = a + b;
  }

  void increment(int a, int b, int &c) const {
    c = a + b;
  }

  // Multiple input with inout param
  void increment(int &a, int b) {
    ++mutableMethodsCalledCount;
    a += b;
  }

  void increment(int &a, int b) const {
    a += b;
  }

  // No input with output (const first)
  int numberOfMutableMethodsCalled() const { return mutableMethodsCalledCount; }
  int numberOfMutableMethodsCalled() { return ++mutableMethodsCalledCount; }

private:
  int mutableMethodsCalledCount = 0;
};

struct HasAmbiguousMethods2 {
  int increment(int a) const {
    return a + 1;
  }
};

struct Unsafe {
  int *ptr;
};

struct HasAmbiguousUnsafeMethods {
  HasAmbiguousUnsafeMethods(const HasAmbiguousUnsafeMethods&);
  Unsafe getUnsafe() const { return Unsafe(); }
  Unsafe getUnsafe() { return Unsafe(); }
};

#endif // TEST_INTEROP_CXX_CLASS_AMBIGUOUS_METHOD_METHODS_H
