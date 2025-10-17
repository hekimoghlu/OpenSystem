/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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
#import "CocoaImage.h"

#import <WebCore/UTIRegistry.h>

#if HAVE(UNIFORM_TYPE_IDENTIFIERS_FRAMEWORK)
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#else
#import <CoreServices/CoreServices.h>
#endif

#import <ImageIO/ImageIO.h>

namespace WebKit {

RetainPtr<NSData> transcode(CGImageRef image, CFStringRef typeIdentifier)
{
    if (!image)
        return nil;

    auto data = adoptNS([[NSMutableData alloc] init]);
    auto destination = adoptCF(CGImageDestinationCreateWithData((__bridge CFMutableDataRef)data.get(), typeIdentifier, 1, nil));
    CGImageDestinationAddImage(destination.get(), image, nil);
    if (!CGImageDestinationFinalize(destination.get()))
        return nil;

    return data;
}

std::pair<RetainPtr<NSData>, RetainPtr<CFStringRef>> transcodeWithPreferredMIMEType(CGImageRef image, CFStringRef preferredMIMEType)
{
    ASSERT(CFStringGetLength(preferredMIMEType));
#if HAVE(UNIFORM_TYPE_IDENTIFIERS_FRAMEWORK)
    auto preferredTypeIdentifier = RetainPtr { (__bridge CFStringRef)[UTType typeWithMIMEType:(__bridge NSString *)preferredMIMEType conformingToType:UTTypeImage].identifier };
#else
    auto preferredTypeIdentifier = adoptCF(UTTypeCreatePreferredIdentifierForTag(kUTTagClassMIMEType, preferredMIMEType, kUTTypeImage));
#endif
    if (WebCore::isSupportedImageType(preferredTypeIdentifier.get())) {
        if (auto data = transcode(image, preferredTypeIdentifier.get()); [data length])
            return { WTFMove(data), WTFMove(preferredTypeIdentifier) };
    }

    return { nil, nil };
}

} // namespace WebKit
