/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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

#include "SVGFitToViewBox.h"
#include "SVGZoomAndPan.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class SVGElement;
class SVGTransformList;
class WeakPtrImplWithEventTargetData;

class SVGViewSpec final : public RefCounted<SVGViewSpec>, public SVGFitToViewBox, public SVGZoomAndPan {
    WTF_MAKE_TZONE_ALLOCATED(SVGViewSpec);
public:
    static Ref<SVGViewSpec> create(SVGElement& contextElement)
    {
        return adoptRef(*new SVGViewSpec(contextElement));
    }

    bool parseViewSpec(StringView);
    void reset();
    void resetContextElement() { m_contextElement = nullptr; }

    RefPtr<SVGElement> viewTarget() const;
    const String& viewTargetString() const { return m_viewTargetString; }

    String transformString() const { return m_transform->valueAsString(); }
    Ref<SVGTransformList>& transform() { return m_transform; }
    Ref<SVGTransformList> protectedTransform();

    const WeakPtr<SVGElement, WeakPtrImplWithEventTargetData>& contextElementConcurrently() const { return m_contextElement; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGViewSpec, SVGFitToViewBox>;

private:
    explicit SVGViewSpec(SVGElement&);

    WeakPtr<SVGElement, WeakPtrImplWithEventTargetData> m_contextElement;
    String m_viewTargetString;
    Ref<SVGTransformList> m_transform;
};

} // namespace WebCore
