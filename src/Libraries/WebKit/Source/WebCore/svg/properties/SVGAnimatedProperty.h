/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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

#include "SVGAttributeAnimator.h"
#include "SVGPropertyOwner.h"
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
    
class SVGElement;

class SVGAnimatedProperty : public ThreadSafeRefCounted<SVGAnimatedProperty>, public SVGPropertyOwner {
public:
    virtual ~SVGAnimatedProperty() = default;
    
    // Manage the relationship with the owner.
    bool isAttached() const { return m_contextElement; }
    void detach() { m_contextElement = nullptr; }
    SVGElement* contextElement() const { return m_contextElement; }
    
    virtual String baseValAsString() const { return emptyString(); }
    virtual String animValAsString() const { return emptyString(); }
    
    // Control the synchronization between the attribute and its reflection in baseVal.
    virtual bool isDirty() const { return false; }
    virtual void setDirty() { }
    virtual std::optional<String> synchronize() { return std::nullopt; }
    
    // Control the animation life cycle.
    bool isAnimating() const { return !m_animators.isEmptyIgnoringNullReferences(); }
    virtual void startAnimation(SVGAttributeAnimator& animator) { m_animators.add(animator); }
    virtual void stopAnimation(SVGAttributeAnimator& animator) { m_animators.remove(animator); }
    
    // Attach/Detach the animVal of the target element's property by the instance element's property.
    virtual void instanceStartAnimation(SVGAttributeAnimator& animator, SVGAnimatedProperty&) { startAnimation(animator); }
    virtual void instanceStopAnimation(SVGAttributeAnimator& animator) { stopAnimation(animator); }
    
protected:
    SVGAnimatedProperty(SVGElement* contextElement)
        : m_contextElement(contextElement)
    {
    }
    
    SVGPropertyOwner* owner() const override;
    void commitPropertyChange(SVGProperty*) override;
    
    SVGElement* m_contextElement { nullptr };
    WeakHashSet<SVGAttributeAnimator> m_animators;
};

} // namespace WebCore

