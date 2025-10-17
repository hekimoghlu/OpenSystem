/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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

#include "IDLTypes.h"
#include "JSDOMConvertBase.h"
#include <wtf/FixedVector.h>

namespace WebCore {

template<typename IDL>
struct VariadicConverter {
    using Item = typename IDL::ImplementationType;

    static std::optional<Item> convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        auto& vm = JSC::getVM(&lexicalGlobalObject);
        auto scope = DECLARE_THROW_SCOPE(vm);

        auto result = WebCore::convert<IDL>(lexicalGlobalObject, value);
        if (UNLIKELY(result.hasException(scope)))
            return std::nullopt;

        return result.releaseReturnValue();
    }
};

template<typename IDL> using VariadicItem = typename VariadicConverter<IDL>::Item;
template<typename IDL> using VariadicArguments = FixedVector<VariadicItem<IDL>>;

template<typename IDL>
VariadicArguments<IDL> convertVariadicArguments(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, size_t startIndex)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto scope = DECLARE_THROW_SCOPE(vm);

    size_t length = callFrame.argumentCount();
    if (startIndex >= length)
        return { };

    auto result = VariadicArguments<IDL>::createWithSizeFromGenerator(length - startIndex, [&](size_t i) -> std::optional<VariadicItem<IDL>> {
        auto result = VariadicConverter<IDL>::convert(lexicalGlobalObject, callFrame.uncheckedArgument(i + startIndex));
        RETURN_IF_EXCEPTION(scope, std::nullopt);

        return result;
    });

    RETURN_IF_EXCEPTION(scope, VariadicArguments<IDL> { });
    return result;
}

} // namespace WebCore
