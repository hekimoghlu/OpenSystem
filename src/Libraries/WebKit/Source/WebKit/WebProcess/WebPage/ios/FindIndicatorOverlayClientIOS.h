/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 9, 2023.
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
#ifndef FindIndicatorOverlayClientIOS_h
#define FindIndicatorOverlayClientIOS_h

#import <WebCore/GraphicsContext.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/PageOverlay.h>
#import <WebCore/TextIndicator.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

class FindIndicatorOverlayClientIOS : public WebCore::PageOverlayClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(FindIndicatorOverlayClientIOS);
public:
    FindIndicatorOverlayClientIOS(WebCore::LocalFrame& frame, WebCore::TextIndicator* textIndicator)
        : m_frame(frame)
        , m_textIndicator(textIndicator)
    {
    }

private:
    void willMoveToPage(WebCore::PageOverlay&, WebCore::Page*) override { }
    void didMoveToPage(WebCore::PageOverlay&, WebCore::Page*) override { }
    void drawRect(WebCore::PageOverlay&, WebCore::GraphicsContext&, const WebCore::IntRect& dirtyRect) override;
    bool mouseEvent(WebCore::PageOverlay&, const WebCore::PlatformMouseEvent&) override { return false; }

    WeakRef<WebCore::LocalFrame> m_frame;
    RefPtr<WebCore::TextIndicator> m_textIndicator;
};

} // namespace WebKit

#endif // FindIndicatorOverlayClientIOS_h
