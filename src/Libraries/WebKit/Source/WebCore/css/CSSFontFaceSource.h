/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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

#include "FontLoadRequest.h"
#include <JavaScriptCore/ArrayBufferView.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class CSSFontFace;
class CSSFontSelector;
class SharedBuffer;
class Document;
class Font;
class FontCreationContext;
class FontDescription;
class SVGFontFaceElement;
class WeakPtrImplWithEventTargetData;

struct FontCustomPlatformData;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(CSSFontFaceSource);
class CSSFontFaceSource final : public FontLoadRequestClient {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(CSSFontFaceSource);
public:
    CSSFontFaceSource(CSSFontFace& owner, AtomString fontFaceName);
    CSSFontFaceSource(CSSFontFace& owner, AtomString fontFaceName, SVGFontFaceElement&);
    CSSFontFaceSource(CSSFontFace& owner, CSSFontSelector&, UniqueRef<FontLoadRequest>);
    CSSFontFaceSource(CSSFontFace& owner, Ref<JSC::ArrayBufferView>&&);
    virtual ~CSSFontFaceSource();

    //                      => Success
    //                    //
    // Pending => Loading
    //                    \\.
    //                      => Failure
    enum class Status {
        Pending,
        Loading,
        Success,
        Failure
    };
    Status status() const { return m_status; }

    void opportunisticallyStartFontDataURLLoading();

    void load(Document* = nullptr);
    RefPtr<Font> font(const FontDescription&, bool syntheticBold, bool syntheticItalic, const FontCreationContext&);

    FontLoadRequest* fontLoadRequest() const { return m_fontRequest.get(); }
    bool requiresExternalResource() const { return m_fontRequest.get(); }

    bool isSVGFontFaceSource() const;

private:
    bool shouldIgnoreFontLoadCompletions() const;

    void fontLoaded(FontLoadRequest&) override;

    void setStatus(Status);

    AtomString m_fontFaceName; // Font name for local fonts
    CSSFontFace& m_face; // Our owning font face.
    WeakPtr<CSSFontSelector> m_fontSelector; // For remote fonts, to orchestrate loading.
    const std::unique_ptr<FontLoadRequest> m_fontRequest; // Also for remote fonts, a pointer to the resource request.

    RefPtr<SharedBuffer> m_generatedOTFBuffer;
    RefPtr<JSC::ArrayBufferView> m_immediateSource;
    RefPtr<FontCustomPlatformData> m_immediateFontCustomPlatformData;

    WeakPtr<SVGFontFaceElement, WeakPtrImplWithEventTargetData> m_svgFontFaceElement;
    RefPtr<FontCustomPlatformData> m_inDocumentCustomPlatformData;

    Status m_status { Status::Pending };
    bool m_hasSVGFontFaceElement { false };
};

} // namespace WebCore
