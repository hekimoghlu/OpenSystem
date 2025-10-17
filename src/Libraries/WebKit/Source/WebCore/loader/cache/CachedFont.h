/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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

#include "CachedResource.h"
#include "CachedResourceClient.h"
#include "Font.h"
#include "FrameLoaderTypes.h"
#include "TextFlags.h"
#include "TrustedFonts.h"
#include <pal/SessionID.h>

namespace WebCore {

class CachedResourceLoader;
class FontCreationContext;
class FontDescription;
class FontPlatformData;
struct FontSelectionSpecifiedCapabilities;
class SVGDocument;
class SVGFontElement;
struct FontCustomPlatformData;

template <typename T> class FontTaggedSettings;
typedef FontTaggedSettings<int> FontFeatureSettings;

class CachedFont : public CachedResource {
public:
    CachedFont(CachedResourceRequest&&, PAL::SessionID, const CookieJar*, Type = Type::FontResource);
    virtual ~CachedFont();

    void beginLoadIfNeeded(CachedResourceLoader&);
    bool stillNeedsLoad() const override { return !m_loadInitiated; }

    virtual bool ensureCustomFontData();
    static RefPtr<FontCustomPlatformData> createCustomFontData(SharedBuffer&, const String& itemInCollection, bool& wrapping);
    static RefPtr<FontCustomPlatformData> createCustomFontDataExperimentalParser(SharedBuffer&, const String& itemInCollection, bool& wrapping);
    static FontPlatformData platformDataFromCustomData(FontCustomPlatformData&, const FontDescription&, bool bold, bool italic, const FontCreationContext&);

    virtual RefPtr<Font> createFont(const FontDescription&, bool syntheticBold, bool syntheticItalic, const FontCreationContext&);

    bool didRefuseToParseCustomFontWithSafeFontParser() const { return m_didRefuseToParseCustomFont; }

protected:
    FontPlatformData platformDataFromCustomData(const FontDescription&, bool bold, bool italic, const FontCreationContext&);

    bool ensureCustomFontData(SharedBuffer* data);

private:
    String calculateItemInCollection() const;

    void checkNotify(const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess = LoadWillContinueInAnotherProcess::No) override;
    bool mayTryReplaceEncodedData() const override;

    void load(CachedResourceLoader&) override;
    NO_RETURN_DUE_TO_ASSERT void setBodyDataFrom(const CachedResource&) final { ASSERT_NOT_REACHED(); }

    void didAddClient(CachedResourceClient&) override;
    void finishLoading(const FragmentedSharedBuffer*, const NetworkLoadMetrics&) override;

    void allClientsRemoved() override;

    FontParsingPolicy policyForCustomFont(const Ref<SharedBuffer>& data);
    void setErrorAndDeleteData();

    bool m_loadInitiated;
    bool m_hasCreatedFontDataWrappingResource;

    FontParsingPolicy m_fontParsingPolicy { FontParsingPolicy::Deny };
    bool m_didRefuseToParseCustomFont { false };

    RefPtr<FontCustomPlatformData> m_fontCustomPlatformData;

    friend class MemoryCache;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CACHED_RESOURCE(CachedFont, CachedResource::Type::FontResource)
