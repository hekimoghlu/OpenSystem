/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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
#import "FontCache.h"

#import "FontCacheCoreText.h"
#import <wtf/cocoa/TypeCastsCocoa.h>

#import <pal/ios/UIKitSoftLink.h>

namespace WebCore {

#if PLATFORM(IOS_FAMILY)
CFStringRef getUIContentSizeCategoryDidChangeNotificationName()
{
    return static_cast<CFStringRef>(PAL::get_UIKit_UIContentSizeCategoryDidChangeNotification());
}
#endif

static String& contentSizeCategoryStorage()
{
    static NeverDestroyed<String> contentSizeCategory;
    return contentSizeCategory.get();
}

CFStringRef contentSizeCategory()
{
    if (!contentSizeCategoryStorage().isNull()) {
        // The contract of this function is that it returns a +0 autoreleased object (just like [[UIApplication sharedApplication] preferredContentSizeCategory] does).
        // String's operator NSString returns a +0 autoreleased object, so we do that here, and then cast it to CFStringRef to return it.
        return bridge_cast(static_cast<NSString*>(contentSizeCategoryStorage()));
    }
#if PLATFORM(IOS_FAMILY)
    return static_cast<CFStringRef>([[PAL::getUIApplicationClass() sharedApplication] preferredContentSizeCategory]);
#else
    return kCTFontContentSizeCategoryL;
#endif
}

void setContentSizeCategory(const String& contentSizeCategory)
{
    if (contentSizeCategory == contentSizeCategoryStorage())
        return;

    contentSizeCategoryStorage() = contentSizeCategory;
    FontCache::invalidateAllFontCaches();
}

} // namespace WebCore
