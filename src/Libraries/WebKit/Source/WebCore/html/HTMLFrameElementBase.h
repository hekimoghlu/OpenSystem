/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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

#include "FrameLoaderTypes.h"
#include "HTMLFrameOwnerElement.h"

namespace JSC {
class CallFrame;
}

namespace WebCore {

class HTMLFrameElementBase : public HTMLFrameOwnerElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLFrameElementBase);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLFrameElementBase);
public:
    void setLocation(JSC::JSGlobalObject&, const String&);

    ScrollbarMode scrollingMode() const final;

protected:
    HTMLFrameElementBase(const QualifiedName&, Document&);

    bool canLoad() const;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void didFinishInsertingNode() final;
    void didAttachRenderers() override;

    WEBCORE_EXPORT void setLocation(const String&);
    void openURL(LockHistory = LockHistory::Yes, LockBackForwardList = LockBackForwardList::Yes);

    AtomString frameURL() const { return m_frameURL; }
    void setFrameURL(const AtomString& frameURL) { m_frameURL = frameURL; }

private:
    bool canLoadScriptURL(const URL&) const final;

    bool canLoadURL(const String& relativeURL) const;
    bool canLoadURL(const URL&) const;

    bool canContainRangeEndPoint() const final { return false; }

    bool supportsFocus() const final;
    void setFocus(bool, FocusVisibility = FocusVisibility::Invisible) final;
    
    bool isURLAttribute(const Attribute&) const final;
    bool isHTMLContentAttribute(const Attribute&) const final;

    AtomString m_frameURL;
    bool m_openingURLAfterInserting { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::HTMLFrameElementBase)
    static bool isType(const WebCore::HTMLElement& element) { return is<WebCore::HTMLFrameElement>(element) || is<WebCore::HTMLIFrameElement>(element); }
    static bool isType(const WebCore::Node& node)
    {
        auto* htmlElement = dynamicDowncast<WebCore::HTMLElement>(node);
        return htmlElement && isType(*htmlElement);
    }
SPECIALIZE_TYPE_TRAITS_END()
