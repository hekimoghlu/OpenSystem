/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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

#if PLATFORM(COCOA)

#include "DictionaryPopupInfo.h"
#include <wtf/Function.h>

#if PLATFORM(MAC)
#import <pal/spi/mac/NSImmediateActionGestureRecognizerSPI.h>
#endif

@class NSView;
@class PDFSelection;
@class UIView;

#if PLATFORM(MAC)
using WKRevealController = id <NSImmediateActionAnimationController>;
using CocoaView = NSView;
#else
using WKRevealController = id;
using CocoaView = UIView;
#endif

namespace WebCore {

class HitTestResult;
class VisibleSelection;

struct SimpleRange;

class DictionaryLookup {
public:
    WEBCORE_EXPORT static std::optional<SimpleRange> rangeForSelection(const VisibleSelection&);
    WEBCORE_EXPORT static std::optional<SimpleRange> rangeAtHitTestResult(const HitTestResult&);
    WEBCORE_EXPORT static NSString *stringForPDFSelection(PDFSelection *);

    // FIXME: Should move/unify dictionaryPopupInfoForRange here too.

    WEBCORE_EXPORT static void showPopup(const DictionaryPopupInfo&, CocoaView *, const Function<void(TextIndicator&)>& textIndicatorInstallationCallback, const Function<FloatRect(FloatRect)>& rootViewToViewConversionCallback = nullptr, Function<void()>&& clearTextIndicator = nullptr);
    WEBCORE_EXPORT static void hidePopup();
    
#if PLATFORM(MAC)
    WEBCORE_EXPORT static WKRevealController animationControllerForPopup(const DictionaryPopupInfo&, NSView *, const Function<void(TextIndicator&)>& textIndicatorInstallationCallback, const Function<FloatRect(FloatRect)>& rootViewToViewConversionCallback = nullptr, Function<void()>&& clearTextIndicator = nullptr);
#endif
};

} // namespace WebCore

#endif // PLATFORM(COCOA)
