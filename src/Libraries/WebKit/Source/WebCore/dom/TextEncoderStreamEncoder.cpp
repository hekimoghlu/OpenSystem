/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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
#include "TextEncoderStreamEncoder.h"

#include <JavaScriptCore/GenericTypedArrayViewInlines.h>
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSGenericTypedArrayViewInlines.h>
#include <wtf/unicode/CharacterNames.h>

namespace WebCore {

RefPtr<Uint8Array> TextEncoderStreamEncoder::encode(const String& input)
{
    StringView view(input);

    if (!view.length())
        return nullptr;

    Vector<uint8_t> bytes(WTF::checkedProduct<size_t>(view.length() + 1, 3));
    auto bytesSpan = bytes.mutableSpan();
    size_t bytesWritten = 0;

    for (size_t cptr = 0; cptr < view.length(); cptr++) {
        // https://encoding.spec.whatwg.org/#convert-code-unit-to-scalar-value
        auto token = view[cptr];
        if (m_pendingLeadSurrogate) {
            auto leadSurrogate = *std::exchange(m_pendingLeadSurrogate, std::nullopt);
            if (U16_IS_TRAIL(token)) {
                auto codePoint = U16_GET_SUPPLEMENTARY(leadSurrogate, token);
                U8_APPEND_UNSAFE(bytesSpan, bytesWritten, codePoint);
                continue;
            }
            U8_APPEND_UNSAFE(bytesSpan, bytesWritten, replacementCharacter);
        }
        if (U16_IS_LEAD(token)) {
            m_pendingLeadSurrogate = token;
            continue;
        }
        if (U16_IS_TRAIL(token)) {
            U8_APPEND_UNSAFE(bytesSpan, bytesWritten, replacementCharacter);
            continue;
        }
        U8_APPEND_UNSAFE(bytesSpan, bytesWritten, token);
    }

    if (!bytesWritten)
        return nullptr;

    bytes.shrink(bytesWritten);
    return Uint8Array::tryCreate(bytes.data(), bytesWritten);
}

RefPtr<Uint8Array> TextEncoderStreamEncoder::flush()
{
    if (!m_pendingLeadSurrogate)
        return nullptr;

    constexpr uint8_t byteSequence[] = { 0xEF, 0xBF, 0xBD };
    return Uint8Array::tryCreate(byteSequence, std::size(byteSequence));
}

}
