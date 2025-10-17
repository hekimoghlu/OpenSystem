/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 23, 2024.
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

#include "FloatRect.h"
#include "LayoutSize.h"
#include "ResizeObserverBoxOptions.h"

#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class Element;
class WeakPtrImplWithEventTargetData;

class ResizeObservation : public RefCounted<ResizeObservation> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ResizeObservation);
public:
    static Ref<ResizeObservation> create(Element& target, ResizeObserverBoxOptions);

    ~ResizeObservation();
    
    struct BoxSizes {
        LayoutSize contentBoxSize;
        LayoutSize contentBoxLogicalSize;
        LayoutSize borderBoxLogicalSize;
    };

    std::optional<BoxSizes> elementSizeChanged() const;
    void updateObservationSize(const BoxSizes&);
    void resetObservationSize();

    FloatRect computeContentRect() const;
    FloatSize borderBoxSize() const;
    FloatSize contentBoxSize() const;
    FloatSize snappedContentBoxSize() const;

    Element* target() const { return m_target.get(); }
    RefPtr<Element> protectedTarget() const;
    ResizeObserverBoxOptions observedBox() const { return m_observedBox; }
    size_t targetElementDepth() const;

private:
    ResizeObservation(Element&, ResizeObserverBoxOptions);

    std::optional<BoxSizes> computeObservedSizes() const;
    LayoutPoint computeTargetLocation() const;

    WeakPtr<Element, WeakPtrImplWithEventTargetData> m_target;
    BoxSizes m_lastObservationSizes;
    ResizeObserverBoxOptions m_observedBox;
};

WTF::TextStream& operator<<(WTF::TextStream&, const ResizeObservation&);

} // namespace WebCore
