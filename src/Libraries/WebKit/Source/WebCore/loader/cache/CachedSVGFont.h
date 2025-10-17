/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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

#include "CachedFont.h"

namespace WebCore {

class FontCreationContext;
class SVGFontFaceElement;
class Settings;
class WeakPtrImplWithEventTargetData;

class CachedSVGFont final : public CachedFont {
public:
    CachedSVGFont(CachedResourceRequest&&, PAL::SessionID, const CookieJar*, const Settings&);
    CachedSVGFont(CachedResourceRequest&&, CachedSVGFont&);
    virtual ~CachedSVGFont();

    bool ensureCustomFontData() final;
    RefPtr<Font> createFont(const FontDescription&, bool syntheticBold, bool syntheticItalic, const FontCreationContext&) final;

private:
    FontPlatformData platformDataFromCustomData(const FontDescription&, bool bold, bool italic, const FontCreationContext&);

    SVGFontElement* getSVGFontById(const AtomString&) const;

    SVGFontElement* maybeInitializeExternalSVGFontElement();
    SVGFontFaceElement* firstFontFace();

    RefPtr<SharedBuffer> m_convertedFont;
    RefPtr<SVGDocument> m_externalSVGDocument;
    WeakPtr<SVGFontElement, WeakPtrImplWithEventTargetData> m_externalSVGFontElement;
    const Ref<const Settings> m_settings;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CACHED_RESOURCE(CachedSVGFont, CachedResource::Type::SVGFontResource)
