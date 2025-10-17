/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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

#if PLATFORM(IOS_FAMILY)

#include "ArgumentCoders.h"
#include <WebCore/IntPoint.h>
#include <WebCore/SelectionGeometry.h>
#include <WebCore/ShareableBitmap.h>
#include <WebCore/TextIndicator.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

struct InteractionInformationRequest {
    WebCore::IntPoint point;

    bool includeSnapshot { false };
    bool includeLinkIndicator { false };
    bool includeCursorContext { false };
    bool includeHasDoubleClickHandler { true };
    bool includeImageData { false };

    bool gatherAnimations { false };
    bool linkIndicatorShouldHaveLegacyMargins { false };

    InteractionInformationRequest() { }
    explicit InteractionInformationRequest(WebCore::IntPoint point)
        : point(point)
    {
    }

    explicit InteractionInformationRequest(WebCore::IntPoint point, bool includeSnapshot, bool includeLinkIndicator, bool includeCursorContext, bool includeHasDoubleClickHandler, bool includeImageData, bool gatherAnimations, bool linkIndicatorShouldHaveLegacyMargins)
        : point(point)
        , includeSnapshot(includeSnapshot)
        , includeLinkIndicator(includeLinkIndicator)
        , includeCursorContext(includeCursorContext)
        , includeHasDoubleClickHandler(includeHasDoubleClickHandler)
        , includeImageData(includeImageData)
        , gatherAnimations(gatherAnimations)
        , linkIndicatorShouldHaveLegacyMargins(linkIndicatorShouldHaveLegacyMargins)
    {
    }

    bool isValidForRequest(const InteractionInformationRequest&, int radius = 0) const;
    bool isApproximatelyValidForRequest(const InteractionInformationRequest&, int radius) const;
};

}

#endif // PLATFORM(IOS_FAMILY)
