/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
#include "ArgumentCoders.h"

#include "DaemonDecoder.h"
#include "DaemonEncoder.h"
#include "StreamConnectionEncoder.h"
#include <wtf/text/AtomString.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

namespace IPC {

template<typename Encoder>
void ArgumentCoder<String>::encode(Encoder& encoder, const String& string)
{
    // Special case the null string.
    if (string.isNull()) {
        encoder << std::numeric_limits<unsigned>::max();
        return;
    }

    bool is8Bit = string.is8Bit();
    encoder << string.length() << is8Bit;

    if (is8Bit)
        encoder.encodeSpan(string.span8());
    else
        encoder.encodeSpan(string.span16());
}
template
void ArgumentCoder<String>::encode<Encoder>(Encoder&, const String&);
template
void ArgumentCoder<String>::encode<StreamConnectionEncoder>(StreamConnectionEncoder&, const String&);

template<typename CharacterType, typename Decoder>
static inline std::optional<String> decodeStringText(Decoder& decoder, unsigned length)
{
    auto data = decoder.template decodeSpan<CharacterType>(length);
    if (!data.data())
        return std::nullopt;
    return std::make_optional<String>(data);
}

template<typename Decoder>
WARN_UNUSED_RETURN std::optional<String> ArgumentCoder<String>::decode(Decoder& decoder)
{
    auto length = decoder.template decode<unsigned>();
    if (!length)
        return std::nullopt;
    
    if (*length == std::numeric_limits<unsigned>::max()) {
        // This is the null string.
        return String();
    }

    auto is8Bit = decoder.template decode<bool>();
    if (!is8Bit)
        return std::nullopt;
    
    if (*is8Bit)
        return decodeStringText<LChar>(decoder, *length);
    return decodeStringText<UChar>(decoder, *length);
}
template
std::optional<String> ArgumentCoder<String>::decode<Decoder>(Decoder&);

template<typename Encoder>
void ArgumentCoder<StringView>::encode(Encoder& encoder, StringView string)
{
    // Special case the null string.
    if (string.isNull()) {
        encoder << std::numeric_limits<uint32_t>::max();
        return;
    }

    unsigned length = string.length();
    bool is8Bit = string.is8Bit();

    encoder << length << is8Bit;

    if (is8Bit)
        encoder.encodeSpan(string.span8());
    else
        encoder.encodeSpan(string.span16());
}
template
void ArgumentCoder<StringView>::encode<Encoder>(Encoder&, StringView);
template
void ArgumentCoder<StringView>::encode<StreamConnectionEncoder>(StreamConnectionEncoder&, StringView);

} // namespace IPC
