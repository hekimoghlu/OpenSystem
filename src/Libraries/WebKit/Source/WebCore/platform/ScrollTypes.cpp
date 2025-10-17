/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
#include "ScrollTypes.h"

#include "ScrollBehavior.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

TextStream& operator<<(TextStream& ts, ScrollType scrollType)
{
    switch (scrollType) {
    case ScrollType::User: ts << "user"; break;
    case ScrollType::Programmatic: ts << "programmatic"; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, NativeScrollbarVisibility scrollBarHidden)
{
    switch (scrollBarHidden) {
    case NativeScrollbarVisibility::Visible: ts << 0; break;
    case NativeScrollbarVisibility::HiddenByStyle: ts << 1; break;
    case NativeScrollbarVisibility::ReplacedByCustomScrollbar: ts << 2; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, ScrollClamping clamping)
{
    switch (clamping) {
    case ScrollClamping::Unclamped: ts << "unclamped"; break;
    case ScrollClamping::Clamped: ts << "clamped"; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, ScrollBehaviorForFixedElements behavior)
{
    switch (behavior) {
    case ScrollBehaviorForFixedElements::StickToDocumentBounds:
        ts << 0;
        break;
    case ScrollBehaviorForFixedElements::StickToViewportBounds:
        ts << 1;
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, ScrollBehavior behavior)
{
    switch (behavior) {
    case ScrollBehavior::Auto: ts << "auto"; break;
    case ScrollBehavior::Instant: ts << "instant"; break;
    case ScrollBehavior::Smooth: ts << "smooth"; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, ScrollElasticity behavior)
{
    switch (behavior) {
    case ScrollElasticity::Automatic:
        ts << 0;
        break;
    case ScrollElasticity::None:
        ts << 1;
        break;
    case ScrollElasticity::Allowed:
        ts << 2;
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, ScrollbarMode behavior)
{
    switch (behavior) {
    case ScrollbarMode::Auto:
        ts << 0;
        break;
    case ScrollbarMode::AlwaysOff:
        ts << 1;
        break;
    case ScrollbarMode::AlwaysOn:
        ts << 2;
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, OverflowAnchor behavior)
{
    switch (behavior) {
    case OverflowAnchor::Auto:
        ts << 0;
        break;
    case OverflowAnchor::None:
        ts << 1;
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, ScrollDirection direction)
{
    switch (direction) {
    case ScrollDirection::ScrollUp:
        ts << "up";
        break;
    case ScrollDirection::ScrollDown:
        ts << "down";
        break;
    case ScrollDirection::ScrollLeft:
        ts << "left";
        break;
    case ScrollDirection::ScrollRight:
        ts << "right";
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, ScrollGranularity granularity)
{
    switch (granularity) {
    case ScrollGranularity::Line:
        ts << "line";
        break;
    case ScrollGranularity::Page:
        ts << "page";
        break;
    case ScrollGranularity::Document:
        ts << "document";
        break;
    case ScrollGranularity::Pixel:
        ts << "pixel";
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, ScrollbarWidth width)
{
    switch (width) {
    case ScrollbarWidth::Auto:
        ts << "auto";
        break;
    case ScrollbarWidth::Thin:
        ts << "thin";
        break;
    case ScrollbarWidth::None:
        ts << "none";
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, ScrollPositionChangeOptions options)
{
    ts.dumpProperty("type", options.type);
    ts.dumpProperty("clamping", options.clamping);
    ts.dumpProperty("animated", options.animated == ScrollIsAnimated::Yes);
    ts.dumpProperty("snap point selection method", options.snapPointSelectionMethod);
    ts.dumpProperty("original scroll delta", options.originalScrollDelta ? *options.originalScrollDelta : FloatSize());

    return ts;
}

TextStream& operator<<(TextStream& ts, ScrollSnapPointSelectionMethod option)
{
    switch (option) {
    case ScrollSnapPointSelectionMethod::Directional:
        ts << "Directional";
        break;
    case ScrollSnapPointSelectionMethod::Closest:
        ts << "Closest";
        break;
    }
    return ts;
}

} // namespace WebCore
