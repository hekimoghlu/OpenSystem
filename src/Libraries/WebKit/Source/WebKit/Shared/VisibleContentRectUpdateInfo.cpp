/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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

#if ENABLE(UI_SIDE_COMPOSITING)

#include "VisibleContentRectUpdateInfo.h"

#include <WebCore/LengthBox.h>
#include <wtf/text/TextStream.h>

namespace WebKit {
using namespace WebCore;

String VisibleContentRectUpdateInfo::dump() const
{
    TextStream stream;
    stream << *this;
    return stream.release();
}

TextStream& operator<<(TextStream& ts, ViewStabilityFlag stabilityFlag)
{
    switch (stabilityFlag) {
    case ViewStabilityFlag::ScrollViewInteracting: ts << "scroll view interacting"; break;
    case ViewStabilityFlag::ScrollViewAnimatedScrollOrZoom: ts << "scroll view animated scroll or zoom"; break;
    case ViewStabilityFlag::ScrollViewRubberBanding: ts << "scroll view rubberbanding"; break;
    case ViewStabilityFlag::ChangingObscuredInsetsInteractively: ts << "changing obscured insets interactively"; break;
    case ViewStabilityFlag::UnstableForTesting: ts << "unstable for testing"; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, const VisibleContentRectUpdateInfo& info)
{
    TextStream::GroupScope scope(ts);
    
    ts << "VisibleContentRectUpdateInfo";

    ts.dumpProperty("lastLayerTreeTransactionID", info.lastLayerTreeTransactionID());

    ts.dumpProperty("exposedContentRect", info.exposedContentRect());
    ts.dumpProperty("unobscuredContentRect", info.unobscuredContentRect());
    ts.dumpProperty("contentInsets", info.contentInsets());
    ts.dumpProperty("unobscuredContentRectRespectingInputViewBounds", info.unobscuredContentRectRespectingInputViewBounds());
    ts.dumpProperty("unobscuredRectInScrollViewCoordinates", info.unobscuredRectInScrollViewCoordinates());
    ts.dumpProperty("layoutViewportRect", info.layoutViewportRect());
    ts.dumpProperty("obscuredInsets", info.obscuredInsets());
    ts.dumpProperty("unobscuredSafeAreaInsets", info.unobscuredSafeAreaInsets());

    ts.dumpProperty("scale", info.scale());
    ts.dumpProperty("viewStability", info.viewStability());
    ts.dumpProperty("isFirstUpdateForNewViewSize", info.isFirstUpdateForNewViewSize());
    if (info.enclosedInScrollableAncestorView())
        ts.dumpProperty("enclosedInScrollableAncestorView", info.enclosedInScrollableAncestorView());

    ts.dumpProperty("allowShrinkToFit", info.allowShrinkToFit());
    ts.dumpProperty("scrollVelocity", info.scrollVelocity());

    return ts;
}

} // namespace WebKit

#endif // ENABLE(UI_SIDE_COMPOSITING)
