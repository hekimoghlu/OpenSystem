/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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
#include "PlatformPasteboard.h"

#if USE(LIBWPE)

#include "Pasteboard.h"
#include <wpe/wpe.h>
#include <wtf/Assertions.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

PlatformPasteboard::PlatformPasteboard(const String&)
    : m_pasteboard(wpe_pasteboard_get_singleton())
{
    ASSERT(m_pasteboard);
}

PlatformPasteboard::PlatformPasteboard()
    : m_pasteboard(wpe_pasteboard_get_singleton())
{
    ASSERT(m_pasteboard);
}

void PlatformPasteboard::performAsDataOwner(DataOwnerType, NOESCAPE Function<void()>&& actions)
{
    actions();
}

void PlatformPasteboard::getTypes(Vector<String>& types) const
{
    struct wpe_pasteboard_string_vector pasteboardTypes = { nullptr, 0 };
    wpe_pasteboard_get_types(m_pasteboard, &pasteboardTypes);
    for (auto& typeString : unsafeMakeSpan(pasteboardTypes.strings, pasteboardTypes.length)) {
        const auto length = std::min(static_cast<size_t>(typeString.length), std::numeric_limits<size_t>::max());
        types.append(String({ typeString.data, length }));
    }

    wpe_pasteboard_string_vector_free(&pasteboardTypes);
}

String PlatformPasteboard::readString(size_t, const String& type) const
{
    struct wpe_pasteboard_string string = { nullptr, 0 };
    wpe_pasteboard_get_string(m_pasteboard, type.utf8().data(), &string);
    if (!string.length)
        return String();

    const auto length = std::min(static_cast<size_t>(string.length), std::numeric_limits<size_t>::max());
    String returnValue({ string.data, length });

    wpe_pasteboard_string_free(&string);
    return returnValue;
}

void PlatformPasteboard::write(const PasteboardWebContent& content)
{
    static constexpr auto plainText = "text/plain;charset=utf-8"_s;
    static constexpr auto htmlText = "text/html;charset=utf-8"_s;

    CString textString = content.text.utf8();
    CString markupString = content.markup.utf8();

    std::array<struct wpe_pasteboard_string_pair, 2> pairs = { {
        { { nullptr, 0 }, { nullptr, 0 } },
        { { nullptr, 0 }, { nullptr, 0 } },
    } };
    wpe_pasteboard_string_initialize(&pairs[0].type, plainText, strlen(plainText));
    wpe_pasteboard_string_initialize(&pairs[0].string, textString.data(), textString.length());
    wpe_pasteboard_string_initialize(&pairs[1].type, htmlText, strlen(htmlText));
    wpe_pasteboard_string_initialize(&pairs[1].string, markupString.data(), markupString.length());
    struct wpe_pasteboard_string_map map = { pairs.data(), pairs.size() };

    wpe_pasteboard_write(m_pasteboard, &map);

    wpe_pasteboard_string_free(&pairs[0].type);
    wpe_pasteboard_string_free(&pairs[0].string);
    wpe_pasteboard_string_free(&pairs[1].type);
    wpe_pasteboard_string_free(&pairs[1].string);
}

void PlatformPasteboard::write(const String& type, const String& string)
{
    struct wpe_pasteboard_string_pair pairs[] = {
        { { nullptr, 0 }, { nullptr, 0 } },
    };

    auto typeUTF8 = type.utf8();
    auto stringUTF8 = string.utf8();
    wpe_pasteboard_string_initialize(&pairs[0].type, typeUTF8.data(), typeUTF8.length());
    wpe_pasteboard_string_initialize(&pairs[0].string, stringUTF8.data(), stringUTF8.length());
    struct wpe_pasteboard_string_map map = { pairs, 1 };

    wpe_pasteboard_write(m_pasteboard, &map);

    wpe_pasteboard_string_free(&pairs[0].type);
    wpe_pasteboard_string_free(&pairs[0].string);
}

Vector<String> PlatformPasteboard::typesSafeForDOMToReadAndWrite(const String&) const
{
    return { };
}

int64_t PlatformPasteboard::write(const PasteboardCustomData&)
{
    return 0;
}

int64_t PlatformPasteboard::write(const Vector<PasteboardCustomData>&)
{
    return 0;
}

} // namespace WebCore

#endif // USE(LIBWPE)
