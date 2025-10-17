/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#import "CoreIPCFont.h"

#if PLATFORM(COCOA)

#import "CoreIPCNSCFObject.h"
#import "CoreIPCTypes.h"
#import "CoreTextHelpers.h"
#import <wtf/BlockObjCExceptions.h>

#if PLATFORM(IOS_FAMILY)
#import <UIKit/UIFont.h>
#import <UIKit/UIFontDescriptor.h>
#endif

namespace WebKit {

CoreIPCFont::CoreIPCFont(WebCore::CocoaFont *font)
    : m_fontDescriptorAttributes(font.fontDescriptor.fontAttributes)
{
}

RetainPtr<id> CoreIPCFont::toID() const
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS

    return { WebKit::fontWithAttributes(m_fontDescriptorAttributes.toID().get(), 0) };

    END_BLOCK_OBJC_EXCEPTIONS

    return nullptr;
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
