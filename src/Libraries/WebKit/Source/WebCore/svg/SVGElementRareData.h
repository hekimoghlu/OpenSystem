/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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

#include "MutableStyleProperties.h"
#include "SVGResourceElementClient.h"
#include "SVGTests.h"
#include "StyleResolver.h"
#include <wtf/HashSet.h>
#include <wtf/Noncopyable.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {

class CSSCursorImageValue;
class SVGCursorElement;
class SVGElement;

class SVGElementRareData {
    WTF_MAKE_TZONE_ALLOCATED(SVGElementRareData);
    WTF_MAKE_NONCOPYABLE(SVGElementRareData);
public:
    SVGElementRareData()
        : m_instancesUpdatesBlocked(false)
        , m_useOverrideComputedStyle(false)
        , m_needsOverrideComputedStyleUpdate(false)
    {
    }

    void addInstance(SVGElement& element) { m_instances.add(element); }
    void removeInstance(SVGElement& element) { m_instances.remove(element); }
    const WeakHashSet<SVGElement, WeakPtrImplWithEventTargetData>& instances() const { return m_instances; }

    bool instanceUpdatesBlocked() const { return m_instancesUpdatesBlocked; }
    void setInstanceUpdatesBlocked(bool value) { m_instancesUpdatesBlocked = value; }

    void addReferencingElement(SVGElement& element) { m_referencingElements.add(element); }
    void removeReferencingElement(SVGElement& element) { m_referencingElements.remove(element); }
    const WeakHashSet<SVGElement, WeakPtrImplWithEventTargetData>& referencingElements() const { return m_referencingElements; }
    WeakHashSet<SVGElement, WeakPtrImplWithEventTargetData> takeReferencingElements() { return std::exchange(m_referencingElements, { }); }
    SVGElement* referenceTarget() const { return m_referenceTarget.get(); }
    void setReferenceTarget(WeakPtr<SVGElement, WeakPtrImplWithEventTargetData>&& element) { m_referenceTarget = WTFMove(element); }

    void addReferencingCSSClient(SVGResourceElementClient& client) { m_referencingCSSClients.add(client); }
    void removeReferencingCSSClient(SVGResourceElementClient& client) { m_referencingCSSClients.remove(client); }
    const WeakHashSet<SVGResourceElementClient>& referencingCSSClients() const { return m_referencingCSSClients; }

    SVGElement* correspondingElement() { return m_correspondingElement.get(); }
    void setCorrespondingElement(SVGElement* correspondingElement) { m_correspondingElement = correspondingElement; }

    MutableStyleProperties* animatedSMILStyleProperties() const { return m_animatedSMILStyleProperties.get(); }
    MutableStyleProperties& ensureAnimatedSMILStyleProperties()
    {
        if (!m_animatedSMILStyleProperties)
            m_animatedSMILStyleProperties = MutableStyleProperties::create(SVGAttributeMode);
        return *m_animatedSMILStyleProperties;
    }

    inline const RenderStyle* overrideComputedStyle(Element&, const RenderStyle* parentStyle);

    bool useOverrideComputedStyle() const { return m_useOverrideComputedStyle; }
    void setUseOverrideComputedStyle(bool value) { m_useOverrideComputedStyle = value; }
    void setNeedsOverrideComputedStyleUpdate() { m_needsOverrideComputedStyleUpdate = true; }

    SVGConditionalProcessingAttributes* conditionalProcessingAttributesIfExists() const { return m_conditionalProcessingAttributes.get(); }
    SVGConditionalProcessingAttributes& conditionalProcessingAttributes(SVGElement& contextElement)
    {
        if (!m_conditionalProcessingAttributes)
            m_conditionalProcessingAttributes = makeUnique<SVGConditionalProcessingAttributes>(contextElement);
        return *m_conditionalProcessingAttributes;
    }

private:
    WeakHashSet<SVGElement, WeakPtrImplWithEventTargetData> m_referencingElements;
    WeakPtr<SVGElement, WeakPtrImplWithEventTargetData> m_referenceTarget;

    WeakHashSet<SVGResourceElementClient> m_referencingCSSClients;

    WeakHashSet<SVGElement, WeakPtrImplWithEventTargetData> m_instances;
    WeakPtr<SVGElement, WeakPtrImplWithEventTargetData> m_correspondingElement;
    bool m_instancesUpdatesBlocked : 1;
    bool m_useOverrideComputedStyle : 1;
    bool m_needsOverrideComputedStyleUpdate : 1;
    RefPtr<MutableStyleProperties> m_animatedSMILStyleProperties;
    std::unique_ptr<RenderStyle> m_overrideComputedStyle;
    std::unique_ptr<SVGConditionalProcessingAttributes> m_conditionalProcessingAttributes;
};

} // namespace WebCore
