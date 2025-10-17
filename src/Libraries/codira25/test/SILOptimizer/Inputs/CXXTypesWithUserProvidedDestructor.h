/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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

#ifndef TEST_SIL_OPTIMIZER_CXX_WITH_CUSTOM_DESTRUCTOR_H
#define TEST_SIL_OPTIMIZER_CXX_WITH_CUSTOM_DESTRUCTOR_H

struct HasUserProvidedDestructor {
  int x = 0;

  HasUserProvidedDestructor(const HasUserProvidedDestructor &other) : x(other.x) {}
  ~HasUserProvidedDestructor() {}
};

struct Loadable {
  int x = 0;
};

struct HasMemberWithUserProvidedDestructor {
  Loadable y;

  HasMemberWithUserProvidedDestructor(const HasMemberWithUserProvidedDestructor &other) : y(other.y) {}
  ~HasMemberWithUserProvidedDestructor() {}
};

void foo();

struct NonCopyable {
  NonCopyable(int x) : x(x) {}
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable(NonCopyable &&other) : x(other.x) { other.x = -123; }
  ~NonCopyable() { foo(); }

  int x;
};

#endif // TEST_SIL_OPTIMIZER_CXX_WITH_CUSTOM_DESTRUCTOR_H
