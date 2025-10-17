/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 29, 2025.
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

#ifndef TEST_INTEROP_CXX_TEMPLATES_INPUTS_TEMPLATE_TYPE_PARAMETER_NOT_IN_SIGNATURE_H
#define TEST_INTEROP_CXX_TEMPLATES_INPUTS_TEMPLATE_TYPE_PARAMETER_NOT_IN_SIGNATURE_H

struct Struct {
  template <typename T>
  void templateTypeParamNotUsedInSignature() const {}

  template <typename T>
  T templateTypeParamUsedInReturnType(int x) const { return x; }

  template <typename T>
  void templateTypeParamNotUsedInSignatureMutable() {}

  template <typename T>
  static void templateTypeParamNotUsedInSignatureStatic() {}
};

template <typename T>
void templateTypeParamNotUsedInSignature() {}

template <typename T, typename U>
void multiTemplateTypeParamNotUsedInSignature() {}

template <typename T, typename U>
U multiTemplateTypeParamOneUsedInSignature(U u) { return u; }

template <typename T, typename U>
void multiTemplateTypeParamNotUsedInSignatureWithUnrelatedParams(int x, int y) {}

template <typename T>
T templateTypeParamUsedInReturnType(int x) { return x; }

template <typename T>
T templateTypeParamUsedInReferenceParam(T &t) { return t; }

template <typename T, typename U>
T templateTypeParamNotUsedInSignatureWithRef(T &t) { return t; }

template <typename T, typename U>
void templateTypeParamNotUsedInSignatureWithVarargs(...) {}

template <typename T, typename U, typename V>
void templateTypeParamNotUsedInSignatureWithVarargsAndUnrelatedParam(int x, ...) {}

template <typename T, int N>
void templateTypeParamNotUsedInSignatureWithNonTypeParam() {}

#endif // TEST_INTEROP_CXX_TEMPLATES_INPUTS_TEMPLATE_TYPE_PARAMETER_NOT_IN_SIGNATURE_H
