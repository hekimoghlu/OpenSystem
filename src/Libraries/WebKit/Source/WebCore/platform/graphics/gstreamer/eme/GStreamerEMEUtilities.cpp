/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
#include "GStreamerEMEUtilities.h"

#include <wtf/StdLibExtras.h>
#include <wtf/text/Base64.h>

#if ENABLE(ENCRYPTED_MEDIA) && USE(GSTREAMER)

GST_DEBUG_CATEGORY_EXTERN(webkit_media_common_encryption_decrypt_debug_category);
#define GST_CAT_DEFAULT webkit_media_common_encryption_decrypt_debug_category

namespace WebCore {

struct GMarkupParseContextUserData {
    bool isParsingPssh { false };
    RefPtr<SharedBuffer> pssh;
};

static void markupStartElement(GMarkupParseContext*, const gchar* elementName, const gchar**, const gchar**, gpointer userDataPtr, GError**)
{
    GMarkupParseContextUserData* userData = static_cast<GMarkupParseContextUserData*>(userDataPtr);
    auto nameView = StringView::fromLatin1(elementName);
    if (nameView.endsWith("pssh"_s))
        userData->isParsingPssh = true;
}

static void markupEndElement(GMarkupParseContext*, const gchar* elementName, gpointer userDataPtr, GError**)
{
    GMarkupParseContextUserData* userData = static_cast<GMarkupParseContextUserData*>(userDataPtr);
    auto nameView = StringView::fromLatin1(elementName);
    if (nameView.endsWith("pssh"_s)) {
        ASSERT(userData->isParsingPssh);
        userData->isParsingPssh = false;
    }
}

static void markupText(GMarkupParseContext*, const gchar* text, gsize textLength, gpointer userDataPtr, GError**)
{
    GMarkupParseContextUserData* userData = static_cast<GMarkupParseContextUserData*>(userDataPtr);
    if (userData->isParsingPssh) {
        auto data = unsafeMakeSpan(reinterpret_cast<const uint8_t*>(text), textLength);
        auto pssh = base64Decode(data);
        if (pssh.has_value())
            userData->pssh = SharedBuffer::create(WTFMove(*pssh));
    }
}

static void markupPassthrough(GMarkupParseContext*, const gchar*, gsize, gpointer, GError**)
{
}

static void markupError(GMarkupParseContext*, GError*, gpointer)
{
}

static GMarkupParser markupParser { markupStartElement, markupEndElement, markupText, markupPassthrough, markupError };

RefPtr<SharedBuffer> InitData::extractCencIfNeeded(RefPtr<SharedBuffer>&& unparsedPayload)
{
    RefPtr<SharedBuffer> payload = WTFMove(unparsedPayload);
    if (!payload || !payload->size())
        return payload;

    GMarkupParseContextUserData userData;
    GUniquePtr<GMarkupParseContext> markupParseContext(g_markup_parse_context_new(&markupParser, (GMarkupParseFlags) 0, &userData, nullptr));

    auto payloadData = spanReinterpretCast<const char>(payload->span());
    if (g_markup_parse_context_parse(markupParseContext.get(), payloadData.data(), payloadData.size(), nullptr)) {
        if (userData.pssh)
            payload = WTFMove(userData.pssh);
        else
            GST_WARNING("XML was parsed but we could not find a viable base64 encoded pssh box");
    }

    return payload;
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA) && USE(GSTREAMER)
