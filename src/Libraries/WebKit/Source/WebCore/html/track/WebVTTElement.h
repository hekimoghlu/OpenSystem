/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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

#if ENABLE(VIDEO)

#include "HTMLElement.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

enum WebVTTNodeType {
    WebVTTNodeTypeNone = 0,
    WebVTTNodeTypeClass,
    WebVTTNodeTypeItalic,
    WebVTTNodeTypeLanguage,
    WebVTTNodeTypeBold,
    WebVTTNodeTypeUnderline,
    WebVTTNodeTypeRuby,
    WebVTTNodeTypeRubyText,
    WebVTTNodeTypeVoice
};

class WebVTTElement final : public Element {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebVTTElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebVTTElement);
public:
    static Ref<Element> create(const WebVTTNodeType, AtomString language, Document&);
    Ref<HTMLElement> createEquivalentHTMLElement(Document&);

    Ref<Element> cloneElementWithoutAttributesAndChildren(TreeScope&);

    void setWebVTTNodeType(WebVTTNodeType type) { m_webVTTNodeType = type; }
    WebVTTNodeType webVTTNodeType() const { return m_webVTTNodeType; }

    bool isPastNode() const { return m_isPastNode; }
    void setIsPastNode(bool value) { m_isPastNode = value; }

    AtomString language() const { return m_language; }
    void setLanguage(const AtomString& value) { m_language = value; }

    static const QualifiedName& voiceAttributeName()
    {
        static NeverDestroyed<QualifiedName> voiceAttr(nullAtom(), "voice"_s, nullAtom());
        return voiceAttr;
    }

    static const QualifiedName& langAttributeName()
    {
        static NeverDestroyed<QualifiedName> voiceAttr(nullAtom(), "lang"_s, nullAtom());
        return voiceAttr;
    }

protected:
    WebVTTElement(WebVTTNodeType, AtomString language, Document&);

    bool isWebVTTElement() const final { return true; }

    bool m_isPastNode { false };
    WebVTTNodeType m_webVTTNodeType;
    AtomString m_language;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::WebVTTElement)
    static bool isType(const WebCore::Node& node) { return node.isWebVTTElement(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(VIDEO)
