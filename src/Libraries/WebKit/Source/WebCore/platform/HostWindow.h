/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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

#include "GraphicsClient.h"
#include "Widget.h"
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(IOS_FAMILY)
OBJC_CLASS NSData;
#endif

namespace WebCore {

class Cursor;

using FramesPerSecond = unsigned;

class HostWindow : public GraphicsClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(HostWindow);
    WTF_MAKE_NONCOPYABLE(HostWindow);
public:
    HostWindow() = default;
    virtual ~HostWindow() = default;

    // Requests the host invalidate the root view, not the contents.
    virtual void invalidateRootView(const IntRect& updateRect) = 0;

    // Requests the host invalidate the contents and the root view.
    virtual void invalidateContentsAndRootView(const IntRect& updateRect) = 0;

    // Requests the host scroll backingstore by the specified delta, rect to scroll, and clip rect.
    virtual void scroll(const IntSize& scrollDelta, const IntRect& rectToScroll, const IntRect& clipRect) = 0;

    // Requests the host invalidate the contents, not the root view. This is the slow path for scrolling.
    virtual void invalidateContentsForSlowScroll(const IntRect& updateRect) = 0;

    // Methods for doing coordinate conversions to and from screen coordinates.
    virtual IntPoint screenToRootView(const IntPoint&) const = 0;
    virtual IntPoint rootViewToScreen(const IntPoint&) const = 0;
    virtual IntRect rootViewToScreen(const IntRect&) const = 0;
    virtual IntPoint accessibilityScreenToRootView(const IntPoint&) const = 0;
    virtual IntRect rootViewToAccessibilityScreen(const IntRect&) const = 0;
#if PLATFORM(IOS_FAMILY)
    virtual void relayAccessibilityNotification(const String&, const RetainPtr<NSData>&) const = 0;
#endif

    // Method for retrieving the native client of the page.
    virtual PlatformPageClient platformPageClient() const = 0;
    
    // Request that the cursor change.
    virtual void setCursor(const Cursor&) = 0;

    virtual void setCursorHiddenUntilMouseMoves(bool) = 0;

    virtual void windowScreenDidChange(PlatformDisplayID, std::optional<FramesPerSecond> nominalFramesPerSecond) = 0;

    virtual FloatSize screenSize() const = 0;
    virtual FloatSize availableScreenSize() const = 0;
    virtual FloatSize overrideScreenSize() const = 0;
    virtual FloatSize overrideAvailableScreenSize() const = 0;
};

} // namespace WebCore
