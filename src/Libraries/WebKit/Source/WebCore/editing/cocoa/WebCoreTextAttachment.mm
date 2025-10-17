/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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
#import "WebCoreTextAttachment.h"

#import <pal/spi/ios/UIKitSPI.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/RetainPtr.h>

#import <pal/ios/UIKitSoftLink.h>

namespace WebCore {

static RetainPtr<CocoaImage>& webCoreTextAttachmentMissingPlatformImageIfExists()
{
    static NeverDestroyed<RetainPtr<CocoaImage>> missingImage;
    return missingImage.get();
}

CocoaImage *webCoreTextAttachmentMissingPlatformImage()
{
    static dispatch_once_t once;

    dispatch_once(&once, ^{
        RetainPtr webCoreBundle = [NSBundle bundleWithIdentifier:@"com.apple.WebCore"];
#if PLATFORM(IOS_FAMILY)
        RetainPtr image = [PAL::getUIImageClass() imageNamed:@"missingImage" inBundle:webCoreBundle.get() compatibleWithTraitCollection:nil];
#else
        RetainPtr image = [webCoreBundle imageForResource:@"missingImage"];
#endif
        ASSERT_WITH_MESSAGE(image != nil, "Unable to find missingImage.");
        webCoreTextAttachmentMissingPlatformImageIfExists() = WTFMove(image);
    });

    return webCoreTextAttachmentMissingPlatformImageIfExists().get();
}

bool isWebCoreTextAttachmentMissingPlatformImage(CocoaImage *image)
{
    return image && image == webCoreTextAttachmentMissingPlatformImageIfExists();
}

} // namespace WebCore
