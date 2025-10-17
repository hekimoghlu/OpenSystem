/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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
#include "config.h"
#include "Widget.h"

#include "FrameView.h"
#include "HostWindow.h"
#include "IntRect.h"
#include "NotImplemented.h"
#include <wtf/Assertions.h>

namespace WebCore {

void Widget::init(PlatformWidget widget)
{
    m_selfVisible = false;
    m_parentVisible = false;
    m_widget = widget;
}

ScrollView* Widget::parent() const
{
    return m_parent.get();
}

RefPtr<ScrollView> Widget::protectedParent() const
{
    return m_parent.get();
}

void Widget::setParent(ScrollView* view)
{
    ASSERT(!view || !m_parent);
    if (!view || !view->isVisible())
        setParentVisible(false);
    m_parent = view;
    if (view && view->isVisible())
        setParentVisible(true);
}

FrameView* Widget::root() const
{
    const Widget* top = this;
    while (top->parent())
        top = top->parent();
    if (auto* frameView = dynamicDowncast<FrameView>(top))
        return const_cast<FrameView*>(frameView);
    return nullptr;
}
    
void Widget::removeFromParent()
{
    if (parent())
        parent()->removeChild(*this);
}

void Widget::setCursor(const Cursor& cursor)
{
    if (auto* view = root())
        view->hostWindow()->setCursor(cursor);
}

IntRect Widget::convertFromRootView(const IntRect& rootRect) const
{
    if (const ScrollView* parentScrollView = parent()) {
        IntRect parentRect = parentScrollView->convertFromRootView(rootRect);
        return convertFromContainingView(parentRect);
    }
    return rootRect;
}

FloatRect Widget::convertFromRootView(const FloatRect& rootRect) const
{
    if (const ScrollView* parentScrollView = parent()) {
        FloatRect parentRect = parentScrollView->convertFromRootView(rootRect);
        return convertFromContainingView(parentRect);
    }
    return rootRect;
}

IntRect Widget::convertToRootView(const IntRect& localRect) const
{
    if (const ScrollView* parentScrollView = parent()) {
        IntRect parentRect = convertToContainingView(localRect);
        return parentScrollView->convertToRootView(parentRect);
    }
    return localRect;
}

FloatRect Widget::convertToRootView(const FloatRect& localRect) const
{
    if (const ScrollView* parentScrollView = parent()) {
        FloatRect parentRect = convertToContainingView(localRect);
        return parentScrollView->convertToRootView(parentRect);
    }
    return localRect;
}

IntPoint Widget::convertFromRootView(const IntPoint& rootPoint) const
{
    if (const ScrollView* parentScrollView = parent()) {
        IntPoint parentPoint = parentScrollView->convertFromRootView(rootPoint);
        return convertFromContainingView(parentPoint);
    }
    return rootPoint;
}

IntPoint Widget::convertToRootView(const IntPoint& localPoint) const
{
    if (const ScrollView* parentScrollView = parent()) {
        IntPoint parentPoint = convertToContainingView(localPoint);
        return parentScrollView->convertToRootView(parentPoint);
    }
    return localPoint;
}

FloatPoint Widget::convertFromRootView(const FloatPoint& rootPoint) const
{
    if (const ScrollView* parentScrollView = parent()) {
        FloatPoint parentPoint = parentScrollView->convertFromRootView(rootPoint);
        return convertFromContainingView(parentPoint);
    }
    return rootPoint;
}

FloatPoint Widget::convertToRootView(const FloatPoint& localPoint) const
{
    if (const ScrollView* parentScrollView = parent()) {
        FloatPoint parentPoint = convertToContainingView(localPoint);
        return parentScrollView->convertToRootView(parentPoint);
    }
    return localPoint;
}

IntRect Widget::convertFromContainingWindow(const IntRect& windowRect) const
{
    if (const ScrollView* parentScrollView = parent()) {
        IntRect parentRect = parentScrollView->convertFromContainingWindow(windowRect);
        return convertFromContainingView(parentRect);
    }
    return convertFromContainingWindowToRoot(this, windowRect);
}

IntRect Widget::convertToContainingWindow(const IntRect& localRect) const
{
    if (const ScrollView* parentScrollView = parent()) {
        IntRect parentRect = convertToContainingView(localRect);
        return parentScrollView->convertToContainingWindow(parentRect);
    }
    return convertFromRootToContainingWindow(this, localRect);
}

IntPoint Widget::convertFromContainingWindow(const IntPoint& windowPoint) const
{
    if (const ScrollView* parentScrollView = parent()) {
        IntPoint parentPoint = parentScrollView->convertFromContainingWindow(windowPoint);
        return convertFromContainingView(parentPoint);
    }
    return convertFromContainingWindowToRoot(this, windowPoint);
}

IntPoint Widget::convertToContainingWindow(const IntPoint& localPoint) const
{
    if (const ScrollView* parentScrollView = parent()) {
        IntPoint parentPoint = convertToContainingView(localPoint);
        return parentScrollView->convertToContainingWindow(parentPoint);
    }
    return convertFromRootToContainingWindow(this, localPoint);
}

IntRect Widget::convertToContainingView(const IntRect& localRect) const
{
    if (const ScrollView* parentScrollView = parent()) {
        IntRect parentRect(localRect);
        parentRect.setLocation(parentScrollView->convertChildToSelf(this, localRect.location()));
        return parentRect;
    }
    return localRect;
}

IntRect Widget::convertFromContainingView(const IntRect& parentRect) const
{
    if (const auto* parentScrollView = parent()) {
        auto localRect = parentRect;
        localRect.setLocation(parentScrollView->convertSelfToChild(this, localRect.location()));
        return localRect;
    }

    return parentRect;
}

FloatRect Widget::convertToContainingView(const FloatRect& localRect) const
{
    return convertToContainingView(IntRect(localRect));
}

FloatRect Widget::convertFromContainingView(const FloatRect& parentRect) const
{
    return convertFromContainingView(IntRect(parentRect));
}

IntPoint Widget::convertToContainingView(const IntPoint& localPoint) const
{
    if (const auto* parentScrollView = parent())
        return parentScrollView->convertChildToSelf(this, localPoint);

    return localPoint;
}

IntPoint Widget::convertFromContainingView(const IntPoint& parentPoint) const
{
    if (const auto* parentScrollView = parent())
        return parentScrollView->convertSelfToChild(this, parentPoint);

    return parentPoint;
}

FloatPoint Widget::convertToContainingView(const FloatPoint& localPoint) const
{
    if (const ScrollView* parentScrollView = parent())
        return parentScrollView->convertChildToSelf(this, localPoint);

    return localPoint;
}

FloatPoint Widget::convertFromContainingView(const FloatPoint& parentPoint) const
{
    return convertFromContainingView(IntPoint(parentPoint));
}

#if !PLATFORM(COCOA)

Widget::Widget(PlatformWidget widget)
{
    init(widget);
}

Widget::~Widget()
{
    ASSERT(!parent());
}

void Widget::setFrameRect(const IntRect& rect)
{
    m_frame = rect;
}

IntRect Widget::frameRect() const
{
    return m_frame;
}

void Widget::show()
{
}

void Widget::hide()
{
}

void Widget::setFocus(bool)
{
}

void Widget::setIsSelected(bool)
{
}

void Widget::paint(GraphicsContext&, const IntRect&, SecurityOriginPaintPolicy, RegionContext*)
{
}

IntRect Widget::convertFromRootToContainingWindow(const Widget*, const IntRect& rect)
{
    return rect;
}

IntRect Widget::convertFromContainingWindowToRoot(const Widget*, const IntRect& rect)
{
    return rect;
}

IntPoint Widget::convertFromRootToContainingWindow(const Widget*, const IntPoint& point)
{
    return point;
}

IntPoint Widget::convertFromContainingWindowToRoot(const Widget*, const IntPoint& point)
{
    return point;
}

#endif // !PLATFORM(COCOA)

} // namespace WebCore
