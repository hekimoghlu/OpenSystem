/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
#include "CachedFont.h"

#include "CachedFontClient.h"
#include "CachedResourceClientWalker.h"
#include "CachedResourceLoader.h"
#include "FontCreationContext.h"
#include "FontCustomPlatformData.h"
#include "FontDescription.h"
#include "FontPlatformData.h"
#include "Logging.h"
#include "MemoryCache.h"
#include "SharedBuffer.h"
#include "SubresourceLoader.h"
#include "TextResourceDecoder.h"
#include "TrustedFonts.h"
#include "TypedElementDescendantIteratorInlines.h"
#include "WOFFFileFormat.h"
#include <pal/crypto/CryptoDigest.h>
#include <wtf/Vector.h>
#include <wtf/text/Base64.h>

namespace WebCore {

CachedFont::CachedFont(CachedResourceRequest&& request, PAL::SessionID sessionID, const CookieJar* cookieJar, Type type)
    : CachedResource(WTFMove(request), type, sessionID, cookieJar)
    , m_loadInitiated(false)
    , m_hasCreatedFontDataWrappingResource(false)
{
}

CachedFont::~CachedFont() = default;

void CachedFont::load(CachedResourceLoader&)
{
    // Don't load the file yet.  Wait for an access before triggering the load.
    setLoading(true);
}

void CachedFont::didAddClient(CachedResourceClient& client)
{
    ASSERT(client.resourceClientType() == CachedFontClient::expectedType());
    if (!isLoading())
        downcast<CachedFontClient>(client).fontLoaded(*this);
}


FontParsingPolicy CachedFont::policyForCustomFont(const Ref<SharedBuffer>& data)
{
    RefPtr loader = m_loader;
    if (!loader)
        return FontParsingPolicy::Deny;

    RefPtr frame = loader->frame();
    if (!frame)
        return FontParsingPolicy::Deny;

    return fontBinaryParsingPolicy(data->span(), frame->settings().downloadableBinaryFontTrustedTypes());
}

void CachedFont::finishLoading(const FragmentedSharedBuffer* data, const NetworkLoadMetrics& metrics)
{
    if (data) {
        Ref dataContiguous = data->makeContiguous();
        m_fontParsingPolicy = policyForCustomFont(dataContiguous);
        if (m_fontParsingPolicy == FontParsingPolicy::Deny) {
            // SafeFontParser failed to parse font, we set a flag to signal it in CachedFontLoadRequest.h
            m_didRefuseToParseCustomFont = true;
            setErrorAndDeleteData();
            return;
        }
        m_data = WTFMove(dataContiguous);
        setEncodedSize(m_data->size());
    } else {
        m_data = nullptr;
        setEncodedSize(0);
    }
    setLoading(false);
    checkNotify(metrics);
}

void CachedFont::setErrorAndDeleteData()
{
    CachedResourceHandle protectedThis { *this };
    setEncodedSize(0);
    error(Status::DecodeError);
    if (inCache())
        MemoryCache::singleton().remove(*this);
    if (RefPtr loader = m_loader)
        loader->cancel();
}

void CachedFont::beginLoadIfNeeded(CachedResourceLoader& loader)
{
    if (!m_loadInitiated) {
        m_loadInitiated = true;
        CachedResource::load(loader);
    }
}

bool CachedFont::ensureCustomFontData()
{
    if (!m_data)
        return ensureCustomFontData(nullptr);
    if (RefPtr data = m_data; !data->isContiguous())
        m_data = data->makeContiguous();
    return ensureCustomFontData(downcast<SharedBuffer>(m_data).get());
}

String CachedFont::calculateItemInCollection() const
{
    return url().fragmentIdentifier().toString();
}

bool CachedFont::ensureCustomFontData(SharedBuffer* data)
{
    if (!m_fontCustomPlatformData && !errorOccurred() && !isLoading() && data) {
        bool wrapping = false;
        switch (m_fontParsingPolicy) {
        case FontParsingPolicy::Deny:
            // This is not supposed to happen: loading should have cancelled
            // back in finishLoading. Nevertheless, we can recover in a healthy
            // manner.
            setErrorAndDeleteData();
            return false;

        case FontParsingPolicy::LoadWithSystemFontParser: {
            m_fontCustomPlatformData = createCustomFontData(*data, calculateItemInCollection(), wrapping);
            if (!m_fontCustomPlatformData)
                RELEASE_LOG(Fonts, "[Font Parser] A font could not be parsed by system font parser.");
            break;
        }
        case FontParsingPolicy::LoadWithSafeFontParser: {
            m_fontCustomPlatformData = createCustomFontDataExperimentalParser(*data, calculateItemInCollection(), wrapping);
            if (!m_fontCustomPlatformData) {
                m_didRefuseToParseCustomFont = true;
                RELEASE_LOG(Fonts, "[Font Parser] A font could not be parsed by safe font parser.");
            }
            break;
        }
        }

        m_hasCreatedFontDataWrappingResource = m_fontCustomPlatformData && wrapping;
        if (!m_fontCustomPlatformData) {
            if (m_fontParsingPolicy == FontParsingPolicy::LoadWithSafeFontParser) {
                m_didRefuseToParseCustomFont = true;
                setErrorAndDeleteData();
            } else
                setStatus(DecodeError);
        }
    }

    return m_fontCustomPlatformData.get();
}

RefPtr<FontCustomPlatformData> CachedFont::createCustomFontData(SharedBuffer& bytes, const String& itemInCollection, bool& wrapping)
{
    RefPtr buffer = { &bytes };
    wrapping = !convertWOFFToSfntIfNecessary(buffer);
    return buffer ? FontCustomPlatformData::create(*buffer, itemInCollection) : nullptr;
}

RefPtr<FontCustomPlatformData> CachedFont::createCustomFontDataExperimentalParser(SharedBuffer& bytes, const String& itemInCollection, bool& wrapping)
{
    RefPtr buffer = { &bytes };
    wrapping = !convertWOFFToSfntIfNecessary(buffer);
    return FontCustomPlatformData::createMemorySafe(*buffer, itemInCollection);
}

RefPtr<Font> CachedFont::createFont(const FontDescription& fontDescription, bool syntheticBold, bool syntheticItalic, const FontCreationContext& fontCreationContext)
{
    return Font::create(platformDataFromCustomData(fontDescription, syntheticBold, syntheticItalic, fontCreationContext), Font::Origin::Remote);
}

FontPlatformData CachedFont::platformDataFromCustomData(const FontDescription& fontDescription, bool bold, bool italic, const FontCreationContext& fontCreationContext)
{
    RefPtr fontCustomPlatformData = m_fontCustomPlatformData;
    ASSERT(fontCustomPlatformData);
    return platformDataFromCustomData(*fontCustomPlatformData, fontDescription, bold, italic, fontCreationContext);
}

FontPlatformData CachedFont::platformDataFromCustomData(FontCustomPlatformData& fontCustomPlatformData, const FontDescription& fontDescription, bool bold, bool italic, const FontCreationContext& fontCreationContext)
{
    return fontCustomPlatformData.fontPlatformData(fontDescription, bold, italic, fontCreationContext);
}

void CachedFont::allClientsRemoved()
{
    m_fontCustomPlatformData = nullptr;
}

void CachedFont::checkNotify(const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess)
{
    if (isLoading())
        return;

    CachedResourceClientWalker<CachedFontClient> walker(*this);
    while (CachedFontClient* client = walker.next())
        client->fontLoaded(*this);
}

bool CachedFont::mayTryReplaceEncodedData() const
{
    // If a FontCustomPlatformData has ever been constructed to wrap the internal resource buffer then it still might be in use somewhere.
    // That platform font object might directly reference the encoded data buffer behind this CachedFont,
    // so replacing it is unsafe.

    return !m_hasCreatedFontDataWrappingResource;
}

}
