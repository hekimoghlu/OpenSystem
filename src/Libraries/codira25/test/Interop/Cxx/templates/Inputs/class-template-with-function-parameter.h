/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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

#ifndef TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_WITH_FUNCTION_ARGUMENT_H
#define TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_WITH_FUNCTION_ARGUMENT_H

template <typename Fn>
class function_wrapper;

template <typename Ret, typename... Params>
class function_wrapper<Ret(Params...)> {
  Ret (*fn)(Params... params) = nullptr;
};

typedef function_wrapper<bool(bool)> FuncBoolToBool;
typedef function_wrapper<bool()> FuncVoidToBool;
typedef function_wrapper<void(bool)> FuncBoolToVoid;
typedef function_wrapper<void()> FuncVoidToVoid;
typedef function_wrapper<int(int)> FuncIntToInt;
typedef function_wrapper<int(int, int)> FuncIntIntToInt;
typedef function_wrapper<void(int, int)> FuncIntIntToVoid;
typedef function_wrapper<void(int, int, bool)> FuncIntIntBoolToVoid;

#endif // TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_WITH_FUNCTION_ARGUMENT_H
