/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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

#include "HTMLTokenizer.h"
#include "SegmentedString.h"
#include <pal/text/TextEncoding.h>
#include <wtf/TZoneMalloc.h>

namespace PAL {
class TextCodec;
}

namespace WebCore {

class HTMLMetaCharsetParser {
    WTF_MAKE_TZONE_ALLOCATED(HTMLMetaCharsetParser);
    WTF_MAKE_NONCOPYABLE(HTMLMetaCharsetParser);
public:
    HTMLMetaCharsetParser();

    // Returns true if done checking, regardless whether an encoding is found.
    bool checkForMetaCharset(std::span<const uint8_t>);

    const PAL::TextEncoding& encoding() { return m_encoding; }

    // The returned encoding might not be valid.
    static PAL::TextEncoding encodingFromMetaAttributes(std::span<const std::pair<StringView, StringView>>);

private:
    bool processMeta(HTMLToken&);

    HTMLTokenizer m_tokenizer;
    const std::unique_ptr<PAL::TextCodec> m_codec;
    SegmentedString m_input;
    bool m_inHeadSection { true };
    bool m_doneChecking { false };
    PAL::TextEncoding m_encoding;
};

} // namespace WebCore
