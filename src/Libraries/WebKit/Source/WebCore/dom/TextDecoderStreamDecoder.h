/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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

#include "TextDecoder.h"

namespace WebCore {

class TextDecoderStreamDecoder : public RefCounted<TextDecoderStreamDecoder> {
public:
    static ExceptionOr<Ref<TextDecoderStreamDecoder>> create(const String& label, bool fatal, bool ignoreBOM);

    String encoding() const { return m_textDecoder->encoding(); }
    ExceptionOr<String> decode(std::optional<BufferSource::VariantType>);
    ExceptionOr<String> flush();

private:
    explicit TextDecoderStreamDecoder(Ref<TextDecoder>&&);
    Ref<TextDecoder> protectedTextDecoder();

    Ref<TextDecoder> m_textDecoder;
};

}
