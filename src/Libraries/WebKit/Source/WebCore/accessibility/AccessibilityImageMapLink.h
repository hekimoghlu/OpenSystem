/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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

#include "AccessibilityMockObject.h"
#include "HTMLAreaElement.h"
#include "HTMLMapElement.h"

namespace WebCore {
    
class AccessibilityImageMapLink final : public AccessibilityMockObject {
public:
    static Ref<AccessibilityImageMapLink> create(AXID);
    virtual ~AccessibilityImageMapLink();
    
    void setHTMLAreaElement(HTMLAreaElement*);
    HTMLAreaElement* areaElement() const { return m_areaElement.get(); }
    
    void setHTMLMapElement(HTMLMapElement* element) { m_mapElement = element; }    
    HTMLMapElement* mapElement() const { return m_mapElement.get(); }
    
    Node* node() const final { return m_areaElement.get(); }

    AccessibilityRole determineAccessibilityRole() final;
    bool isEnabled() const final { return true; }

    Element* anchorElement() const final;
    Element* actionElement() const final;
    URL url() const final;
    String title() const final;
    String description() const final;
    AccessibilityObject* parentObject() const final;

    LayoutRect elementRect() const final;

private:
    explicit AccessibilityImageMapLink(AXID);

    void detachFromParent() final;
    Path elementPath() const final;
    RenderElement* imageMapLinkRenderer() const;
    void accessibilityText(Vector<AccessibilityText>&) const final;
    bool isImageMapLink() const final { return true; }
    bool supportsPath() const final { return true; }

    WeakPtr<HTMLAreaElement, WeakPtrImplWithEventTargetData> m_areaElement;
    WeakPtr<HTMLMapElement, WeakPtrImplWithEventTargetData> m_mapElement;
};
    
} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilityImageMapLink)
    static bool isType(const WebCore::AXCoreObject& object)
    {
        auto* accessibilityObject = dynamicDowncast<WebCore::AccessibilityObject>(object);
        return accessibilityObject && accessibilityObject->isImageMapLink();
    }
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isImageMapLink(); }
SPECIALIZE_TYPE_TRAITS_END()
