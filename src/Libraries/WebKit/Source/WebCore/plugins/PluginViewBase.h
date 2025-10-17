/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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

#include "PlatformLayer.h"
#include "ScrollTypes.h"
#include "Widget.h"

#if PLATFORM(COCOA)
typedef struct objc_object* id;
#endif

namespace WebCore {

class Element;
class GraphicsLayer;
class ScrollableArea;
class Scrollbar;
class VoidCallback;

enum class PluginLayerHostingStrategy : uint8_t {
    None,
    PlatformLayer,
    GraphicsLayer
};

// FIXME: Move these virtual functions all into the Widget class and get rid of this class.
class PluginViewBase : public Widget {
public:
    virtual PluginLayerHostingStrategy layerHostingStrategy() const { return PluginLayerHostingStrategy::None; }
    virtual PlatformLayer* platformLayer() const { return nullptr; }
    virtual GraphicsLayer* graphicsLayer() const { return nullptr; }

    virtual void layerHostingStrategyDidChange() { }

    virtual bool scroll(ScrollDirection, ScrollGranularity) { return false; }
    virtual ScrollPosition scrollPositionForTesting() const { return { }; }

    virtual Scrollbar* horizontalScrollbar() { return nullptr; }
    virtual Scrollbar* verticalScrollbar() { return nullptr; }

    virtual bool wantsWheelEvents() { return false; }
    virtual bool shouldAllowNavigationFromDrags() const { return false; }
    virtual void willDetachRenderer() { }

    virtual ScrollableArea* scrollableArea() const { return nullptr; }
    virtual bool usesAsyncScrolling() const { return false; }
    virtual std::optional<ScrollingNodeID> scrollingNodeID() const { return std::nullopt; }
    virtual void willAttachScrollingNode() { }
    virtual void didAttachScrollingNode() { }

#if PLATFORM(COCOA)
    virtual id accessibilityAssociatedPluginParentForElement(Element*) const { return nullptr; }
#endif
    virtual void setPDFDisplayModeForTesting(const String&) { }
    virtual bool sendEditingCommandToPDFForTesting(const String&, const String&) { return false; }
    virtual Vector<FloatRect> pdfAnnotationRectsForTesting() const { return { }; }
    virtual void unlockPDFDocumentForTesting(const String&) { }
    virtual void setPDFTextAnnotationValueForTesting(unsigned /* pageIndex */, unsigned /* annotationIndex */, const String& /* value */) { };

    virtual void releaseMemory() { }

protected:
    explicit PluginViewBase(PlatformWidget widget = 0) : Widget(widget) { }

private:
    bool isPluginViewBase() const final { return true; }

    friend class Internals;
    virtual void registerPDFTestCallback(RefPtr<VoidCallback>&&) { };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_WIDGET(PluginViewBase, isPluginViewBase())
