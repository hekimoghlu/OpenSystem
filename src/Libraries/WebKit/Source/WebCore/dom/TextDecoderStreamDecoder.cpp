/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
#include "TextDecoderStreamDecoder.h"

namespace WebCore {

ExceptionOr<Ref<TextDecoderStreamDecoder>> TextDecoderStreamDecoder::create(const String& label, bool fatal, bool ignoreBOM)
{
    auto textDecoder = TextDecoder::create(label, { fatal, ignoreBOM });
    if (textDecoder.hasException())
        return textDecoder.releaseException();

    return adoptRef(*new TextDecoderStreamDecoder(textDecoder.releaseReturnValue()));
}

TextDecoderStreamDecoder::TextDecoderStreamDecoder(Ref<TextDecoder>&& textDecoder)
    : m_textDecoder(WTFMove(textDecoder))
{
}

ExceptionOr<String> TextDecoderStreamDecoder::decode(std::optional<BufferSource::VariantType> value)
{
    return protectedTextDecoder()->decode(WTFMove(value), { true });
}

ExceptionOr<String> TextDecoderStreamDecoder::flush()
{
    return protectedTextDecoder()->decode({ }, { false });
}

Ref<TextDecoder> TextDecoderStreamDecoder::protectedTextDecoder()
{
    return m_textDecoder;
}

}
