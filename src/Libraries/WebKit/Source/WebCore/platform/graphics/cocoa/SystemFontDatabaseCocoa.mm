/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
#import "SystemFontDatabaseCoreText.h"

#import <pal/ios/UIKitSoftLink.h>

namespace WebCore {

static auto cocoaFontClass()
{
#if PLATFORM(IOS_FAMILY)
    return PAL::getUIFontClass();
#else
    return NSFont.class;
#endif
};

RetainPtr<CTFontDescriptorRef> SystemFontDatabaseCoreText::smallCaptionFontDescriptor()
{
    auto font = [cocoaFontClass() systemFontOfSize:[cocoaFontClass() smallSystemFontSize]];
    return static_cast<CTFontDescriptorRef>(font.fontDescriptor);
}

RetainPtr<CTFontDescriptorRef> SystemFontDatabaseCoreText::menuFontDescriptor()
{
    return adoptCF(CTFontDescriptorCreateForUIType(kCTFontUIFontMenuItem, [cocoaFontClass() systemFontSize], nullptr));
}

RetainPtr<CTFontDescriptorRef> SystemFontDatabaseCoreText::statusBarFontDescriptor()
{
    return adoptCF(CTFontDescriptorCreateForUIType(kCTFontUIFontSystem, [cocoaFontClass() labelFontSize], nullptr));
}

RetainPtr<CTFontDescriptorRef> SystemFontDatabaseCoreText::miniControlFontDescriptor()
{
#if PLATFORM(IOS_FAMILY)
    return adoptCF(CTFontDescriptorCreateForUIType(kCTFontUIFontMiniSystem, 0, nullptr));
#else
    auto font = [cocoaFontClass() systemFontOfSize:[cocoaFontClass() systemFontSizeForControlSize:NSControlSizeMini]];
    return static_cast<CTFontDescriptorRef>(font.fontDescriptor);
#endif
}

RetainPtr<CTFontDescriptorRef> SystemFontDatabaseCoreText::smallControlFontDescriptor()
{
#if PLATFORM(IOS_FAMILY)
    return adoptCF(CTFontDescriptorCreateForUIType(kCTFontUIFontSmallSystem, 0, nullptr));
#else
    auto font = [cocoaFontClass() systemFontOfSize:[cocoaFontClass() systemFontSizeForControlSize:NSControlSizeSmall]];
    return static_cast<CTFontDescriptorRef>(font.fontDescriptor);
#endif
}

RetainPtr<CTFontDescriptorRef> SystemFontDatabaseCoreText::controlFontDescriptor()
{
#if PLATFORM(IOS_FAMILY)
    return adoptCF(CTFontDescriptorCreateForUIType(kCTFontUIFontSystem, 0, nullptr));
#else
    auto font = [cocoaFontClass() systemFontOfSize:[cocoaFontClass() systemFontSizeForControlSize:NSControlSizeRegular]];
    return static_cast<CTFontDescriptorRef>(font.fontDescriptor);
#endif
}

} // namespace WebCore
