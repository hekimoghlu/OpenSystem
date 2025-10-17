/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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
#import "WebCoreURLResponseIOS.h"

#if PLATFORM(IOS_FAMILY)

#import "QuickLook.h"
#import <MobileCoreServices/MobileCoreServices.h>

#import <pal/ios/QuickLookSoftLink.h>

namespace WebCore {

static inline bool shouldPreferTextPlainMIMEType(const String& mimeType, const String& proposedMIMEType)
{
    return ("text/plain"_s == mimeType) && ((proposedMIMEType == "text/xml"_s) || (proposedMIMEType == "application/xhtml+xml"_s) || (proposedMIMEType == "application/xml"_s) || (proposedMIMEType == "image/svg+xml"_s));
}

void adjustMIMETypeIfNecessary(CFURLResponseRef response, IsMainResourceLoad isMainResourceLoad, IsNoSniffSet isNoSniffSet)
{
    auto type = CFURLResponseGetMIMEType(response);
    if (!type) {
        // FIXME: <rdar://problem/46332893> is fixed, but for some reason, this special case is still needed; should resolve that issue and remove this.
        if (auto extension = filePathExtension(response)) {
            if (CFStringCompare(extension.get(), CFSTR("mjs"), kCFCompareCaseInsensitive) == kCFCompareEqualTo) {
                CFURLResponseSetMIMEType(response, CFSTR("text/javascript"));
                return;
            }
        }
    }

#if !USE(QUICK_LOOK)
    UNUSED_PARAM(isMainResourceLoad);
    UNUSED_PARAM(isNoSniffSet);
#else
    // Ensure that the MIME type is correct so that QuickLook's web plug-in is called when needed.
    // The shouldUseQuickLookForMIMEType function filters out the common MIME types so we don't do unnecessary work in those cases.
    if (isMainResourceLoad == IsMainResourceLoad::Yes && isNoSniffSet == IsNoSniffSet::No && shouldUseQuickLookForMIMEType((__bridge NSString *)type)) {
        RetainPtr<CFStringRef> updatedType;
        auto suggestedFilename = adoptCF(CFURLResponseCopySuggestedFilename(response));
        if (auto quickLookType = adoptNS(PAL::softLink_QuickLook_QLTypeCopyBestMimeTypeForFileNameAndMimeType((__bridge NSString *)suggestedFilename.get(), (__bridge NSString *)type)))
            updatedType = (__bridge CFStringRef)quickLookType.get();
        else if (auto extension = filePathExtension(response))
            updatedType = preferredMIMETypeForFileExtensionFromUTType(extension.get());
        if (updatedType && !shouldPreferTextPlainMIMEType(type, updatedType.get()) && (!type || CFStringCompare(type, updatedType.get(), kCFCompareCaseInsensitive) != kCFCompareEqualTo)) {
            CFURLResponseSetMIMEType(response, updatedType.get());
            return;
        }
    }
#endif // USE(QUICK_LOOK)
    if (!type)
        CFURLResponseSetMIMEType(response, CFSTR("application/octet-stream"));
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
