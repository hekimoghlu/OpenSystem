/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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
#include "ScrollbarTrackCornerSystemImageMac.h"

#if USE(APPKIT)

#import "FloatRect.h"
#import "GraphicsContext.h"
#import "LocalCurrentGraphicsContext.h"
#import <pal/spi/mac/CoreUISPI.h>
#import <pal/spi/mac/NSAppearanceSPI.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollbarTrackCornerSystemImageMac);

Ref<ScrollbarTrackCornerSystemImageMac> ScrollbarTrackCornerSystemImageMac::create()
{
    return adoptRef(*new ScrollbarTrackCornerSystemImageMac());
}

Ref<ScrollbarTrackCornerSystemImageMac> ScrollbarTrackCornerSystemImageMac::create(WebCore::Color&& tintColor, bool useDarkAppearance)
{
    auto result = create();
    result->setTintColor(WTFMove(tintColor));
    result->setUseDarkAppearance(useDarkAppearance);
    return result;
}

ScrollbarTrackCornerSystemImageMac::ScrollbarTrackCornerSystemImageMac()
    : AppKitControlSystemImage(AppKitControlSystemImageType::ScrollbarTrackCorner)
{
}

void ScrollbarTrackCornerSystemImageMac::drawControl(GraphicsContext& graphicsContext, const FloatRect& rect) const
{
    LocalCurrentGraphicsContext localContext(graphicsContext);

    auto cornerDrawingOptions = @{ (__bridge NSString *)kCUIWidgetKey: (__bridge NSString *)kCUIWidgetScrollBarTrackCorner,
        (__bridge NSString *)kCUIIsFlippedKey: (__bridge NSNumber *)kCFBooleanTrue };
    [[NSAppearance currentDrawingAppearance] _drawInRect:rect context:localContext.cgContext() options:cornerDrawingOptions];
}

} // namespace WebCore

#endif // USE(APPKIT)
