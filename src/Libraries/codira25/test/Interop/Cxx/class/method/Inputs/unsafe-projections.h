/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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

#ifndef TEST_INTEROP_CXX_CLASS_METHOD_UNSAFE_PROJECTIONS_H
#define TEST_INTEROP_CXX_CLASS_METHOD_UNSAFE_PROJECTIONS_H

#include <string>

struct NestedSelfContained;
struct Empty;
struct SelfContained;
struct ExplicitSelfContained;
struct NestedExplicitSelfContained;

struct View {
  void *ptr;
  
  void *data() const;
  void *empty() const;
  std::string name() const;
  NestedSelfContained nested() const;
  ExplicitSelfContained explicitSelfContained() const;
  NestedExplicitSelfContained explicitNested() const;
};

struct SelfContained {
  void *ptr;
  SelfContained(const SelfContained&);
  
  std::string name() const;
  SelfContained selfContained() const;
  NestedSelfContained nested() const;
  Empty empty() const;
  int value() const;
  View view() const;
  int *pointer() const;
  ExplicitSelfContained explicitSelfContained() const;
  NestedExplicitSelfContained explicitNested() const;
};

struct NestedSelfContained {
  SelfContained member;
  
  std::string name() const;
  SelfContained selfContained() const;
  NestedSelfContained nested() const;
  Empty empty() const;
  int value() const;
  View view() const;
  int *pointer() const;
  ExplicitSelfContained explicitSelfContained() const;
  NestedExplicitSelfContained explicitNested() const;
};

struct InheritSelfContained: SelfContained {
  std::string name() const;
  SelfContained selfContained() const;
  NestedSelfContained nested() const;
  Empty empty() const;
  int value() const;
  View view() const;
  int *pointer() const;
};

struct __attribute__((language_attr("import_owned"))) ExplicitSelfContained {
  void *ptr;
  
  void *pointer() const;
  View view() const;
  NestedSelfContained nested() const;
};

struct NestedExplicitSelfContained {
  ExplicitSelfContained m;

  SelfContained selfContained() const;
  NestedSelfContained nested() const;
  int value() const;
  View view() const;
  int *pointer() const;
};

struct Empty {
  Empty empty() const;
  void *pointer() const;
  SelfContained selfContained() const;
};

struct IntPair {
  int a; int b;

  int first() const;
  void *pointer() const;
  SelfContained selfContained() const;
};

#endif // TEST_INTEROP_CXX_CLASS_METHOD_UNSAFE_PROJECTIONS_H
