/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
#import "CoreIPCNSShadow.h"

#if PLATFORM(COCOA)

#import <WebCore/AttributedString.h>

#if USE(APPKIT)
#import <AppKit/NSShadow.h>
#endif
#if PLATFORM(IOS_FAMILY)
#import <UIKit/NSShadow.h>
#import <pal/ios/UIKitSoftLink.h>
#endif

namespace WebKit {

CoreIPCNSShadow::CoreIPCNSShadow(NSShadow *shadow)
    : m_shadowOffset(shadow.shadowOffset)
    , m_shadowBlurRadius(shadow.shadowBlurRadius)
    , m_shadowColor(shadow.shadowColor)
{
}

RetainPtr<id> CoreIPCNSShadow::toID() const
{
    RetainPtr<NSShadow> result = adoptNS([PlatformNSShadow new]);
    [result setShadowOffset:m_shadowOffset];
    [result setShadowBlurRadius:m_shadowBlurRadius];
    [result setShadowColor:m_shadowColor.get()];
    return result;
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
