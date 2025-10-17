/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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
#include "config.h"
#include "TextEncoder.h"

#include <JavaScriptCore/GenericTypedArrayViewInlines.h>
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSGenericTypedArrayViewInlines.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {

RefPtr<Uint8Array> TextEncoder::encode(String&& input) const
{
    auto result = input.tryGetUTF8([&](std::span<const char8_t> span) -> RefPtr<Uint8Array> {
        return Uint8Array::tryCreate(byteCast<uint8_t>(span));
    });
    if (result)
        return result.value();
    return Uint8Array::tryCreate(nullptr, 0);
}

auto TextEncoder::encodeInto(String&& input, Ref<Uint8Array>&& array) -> EncodeIntoResult
{
    auto destinationBytes = array->mutableSpan();

    uint64_t read = 0;
    uint64_t written = 0;

    for (auto token : StringView(input).codePoints()) {
        if (written >= destinationBytes.size()) {
            ASSERT(written == destinationBytes.size());
            break;
        }
        UBool sawError = false;
        U8_APPEND(destinationBytes, written, destinationBytes.size(), token, sawError);
        if (sawError)
            break;
        if (U_IS_BMP(token))
            ++read;
        else
            read += 2;
    }

    return { read, written };
}

}
