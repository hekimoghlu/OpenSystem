/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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

#ifndef TEST_INTEROP_CXX_TEMPLATES_INPUTS_FULLY_PRE_DEFINED_CLASS_TEMPLATE_H
#define TEST_INTEROP_CXX_TEMPLATES_INPUTS_FULLY_PRE_DEFINED_CLASS_TEMPLATE_H

template<class T>
struct MagicWrapper {
  T t;
  int getValuePlusArg(int arg) const { return t.getValue() + arg; }
};

struct IntWrapper {
  int value;
  int getValue() const { return value; }
};

inline int forceInstantiation() {
  auto t = MagicWrapper<IntWrapper>();
  return t.getValuePlusArg(14);
}

// The ClassTemplateSpecializationDecl node for MagicWrapper<IntWrapper> already has a definition
// because function above forced the instantiation. Its members are fully
// instantiated, so nothing needs to be explicitly instantiated by the Codira
// compiler.
typedef MagicWrapper<IntWrapper> FullyPreDefinedMagicallyWrappedInt;

#endif // TEST_INTEROP_CXX_TEMPLATES_INPUTS_FULLY_PRE_DEFINED_CLASS_TEMPLATE_H
