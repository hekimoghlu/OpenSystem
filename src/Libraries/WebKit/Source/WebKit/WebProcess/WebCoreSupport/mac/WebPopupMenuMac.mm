/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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
#import "config.h"
#import "WebPopupMenu.h"

#import "PlatformPopupMenuData.h"
#import <WebCore/LocalFrame.h>
#import <WebCore/LocalFrameView.h>
#import <WebCore/PopupMenuClient.h>

namespace WebKit {
using namespace WebCore;

void WebPopupMenu::setUpPlatformData(const IntRect&, PlatformPopupMenuData& data)
{
#if USE(APPKIT)
    // FIXME: font will be nil here for custom fonts, we should fix that.
    CTFontRef font = m_popupClient->menuStyle().font().primaryFont().getCTFont();
    if (!font)
        return;

    auto fontDescriptor = adoptCF(CTFontCopyFontDescriptor(font));
    if (!fontDescriptor)
        return;

    auto attributes = adoptCF(CTFontDescriptorCopyAttributes(fontDescriptor.get()));
    if (!attributes)
        return;
    
    data.fontInfo.fontAttributeDictionary = attributes.get();
    data.shouldPopOver = m_popupClient->shouldPopOver();
    data.hideArrows = !m_popupClient->menuStyle().hasDefaultAppearance();
    data.menuSize = m_popupClient->menuStyle().menuSize();
#else
    UNUSED_PARAM(data);
#endif
}

} // namespace WebKit
