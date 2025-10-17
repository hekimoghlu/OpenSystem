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

#pragma once

// Ensure c++ features are used.
namespace ns {
class T {};
class NamespacedType {};

T *doMakeT();
} // namespace ns

struct Basic {
  int a;
  ns::T *b;
};

Basic makeA();

ns::T* makeT();
void useT(ns::T* v);

using namespacedT = ns::T;
using ns::NamespacedType;

class Methods {
 public:
  virtual ~Methods();

  int SimpleMethod(int);

  int SimpleConstMethod(int) const;
  int some_value;

  static int SimpleStaticMethod(int);
};

class Methods2 {
public:
  int SimpleMethod(int);
};

enum __attribute__((enum_extensibility(open))) OpenEmptyEnum : char {};
