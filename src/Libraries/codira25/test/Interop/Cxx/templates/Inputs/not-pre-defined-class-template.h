/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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

#ifndef TEST_INTEROP_CXX_TEMPLATES_INPUTS_NOT_PRE_DEFINED_CLASS_TEMPLATE_H
#define TEST_INTEROP_CXX_TEMPLATES_INPUTS_NOT_PRE_DEFINED_CLASS_TEMPLATE_H

template<class T>
struct MagicWrapper {
  T t;
  int getValuePlusArg(int arg) const { return t.getValue() + arg; }
};

struct IntWrapper {
  int value;
  int getValue() const { return value; }
};

// The ClassTemplateSpecializationDecl node for MagicWrapper<IntWrapper> doesn't have a
// definition in Clang because nothing in this header required the
// instantiation. Therefore, the Codira compiler must trigger instantiation.
typedef MagicWrapper<IntWrapper> MagicallyWrappedIntWithoutDefinition;

#endif // TEST_INTEROP_CXX_TEMPLATES_INPUTS_NOT_PRE_DEFINED_CLASS_TEMPLATE_H
