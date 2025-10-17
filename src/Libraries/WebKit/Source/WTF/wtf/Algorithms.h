/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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

#include <cstring>
#include <type_traits>
#include <wtf/Assertions.h>
#include <wtf/Compiler.h>

namespace WTF {

template<typename ContainerType, typename AnyOfFunction>
bool anyOf(ContainerType&& container, NOESCAPE AnyOfFunction&& anyOfFunction)
{
    for (auto& value : container) {
        if (anyOfFunction(value))
            return true;
    }
    return false;
}

template<typename ContainerType, typename AllOfFunction>
bool allOf(ContainerType&& container, NOESCAPE AllOfFunction&& allOfFunction)
{
    for (auto& value : container) {
        if (!allOfFunction(value))
            return false;
    }
    return true;
}

} // namespace WTF
