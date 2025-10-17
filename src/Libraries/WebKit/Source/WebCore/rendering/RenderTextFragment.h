/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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

#include "RenderBoxModelObject.h"
#include "RenderText.h"

namespace WebCore {

// Used to represent a text substring of an element, e.g., for text runs that are split because of
// first letter and that must therefore have different styles (and positions in the render tree).
// We cache offsets so that text transformations can be applied in such a way that we can recover
// the original unaltered string from our corresponding DOM node.
class RenderTextFragment final : public RenderText {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderTextFragment);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderTextFragment);
public:
    RenderTextFragment(Text&, const String&, int startOffset, int length);
    RenderTextFragment(Document&, const String&, int startOffset, int length);
    RenderTextFragment(Document&, const String&);

    virtual ~RenderTextFragment();

    bool canBeSelectionLeaf() const override;

    unsigned start() const { return m_start; }
    unsigned end() const { return m_end; }

    RenderBoxModelObject* firstLetter() const { return m_firstLetter.get(); }
    void setFirstLetter(RenderBoxModelObject& firstLetter) { m_firstLetter = firstLetter; }
    
    RenderBlock* blockForAccompanyingFirstLetter();

    void setContentString(const String& text);
    StringImpl* contentString() const { return m_contentString.impl(); }

    const String& altText() const { return m_altText; }
    void setAltText(const String& altText) { m_altText = altText; }
    
private:
    void setTextInternal(const String&, bool force) override;

    Vector<UChar> previousCharacter() const override;

    unsigned m_start;
    unsigned m_end;
    // Alternative description that can be used for accessibility instead of the native text.
    String m_altText;
    String m_contentString;
    SingleThreadWeakPtr<RenderBoxModelObject> m_firstLetter;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::RenderTextFragment)
    static bool isType(const WebCore::RenderText& renderer) { return renderer.isRenderTextFragment(); }
    static bool isType(const WebCore::RenderObject& renderer)
    {
        auto* text = dynamicDowncast<WebCore::RenderText>(renderer);
        return text && isType(*text);
    }
SPECIALIZE_TYPE_TRAITS_END()
