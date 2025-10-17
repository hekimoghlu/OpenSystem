/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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
#import "PasteboardTypes.h"

#import <WebCore/LegacyNSPasteboardTypes.h>
#import <wtf/RetainPtr.h>

#if PLATFORM(MAC)

namespace WebKit {

NSString * const PasteboardTypes::WebArchivePboardType = @"Apple Web Archive pasteboard type";
NSString * const PasteboardTypes::WebURLsWithTitlesPboardType = @"WebURLsWithTitlesPboardType";
NSString * const PasteboardTypes::WebURLPboardType = @"public.url";
NSString * const PasteboardTypes::WebURLNamePboardType = @"public.url-name";
NSString * const PasteboardTypes::WebDummyPboardType = @"Apple WebKit dummy pasteboard type";
    
NSArray* PasteboardTypes::forEditing()
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN 
    static NeverDestroyed<RetainPtr<NSArray>> types = @[WebArchivePboardType, (__bridge NSString *)kUTTypeWebArchive, WebCore::legacyHTMLPasteboardType(), WebCore::legacyFilenamesPasteboardType(), WebCore::legacyTIFFPasteboardType(), WebCore::legacyPDFPasteboardType(),
        WebCore::legacyURLPasteboardType(), WebCore::legacyRTFDPasteboardType(), WebCore::legacyRTFPasteboardType(), WebCore::legacyStringPasteboardType(), WebCore::legacyColorPasteboardType(), (__bridge NSString *)kUTTypePNG];
ALLOW_DEPRECATED_DECLARATIONS_END
    return types.get().get();
}

NSArray* PasteboardTypes::forURL()
{
    static NeverDestroyed<RetainPtr<NSArray>> types = @[WebURLsWithTitlesPboardType, WebCore::legacyURLPasteboardType(), WebURLPboardType,  WebURLNamePboardType, WebCore::legacyStringPasteboardType(), WebCore::legacyFilenamesPasteboardType(), WebCore::legacyFilesPromisePasteboardType()];
    return types.get().get();
}

NSArray* PasteboardTypes::forImages()
{
    static NeverDestroyed<RetainPtr<NSArray>> types = @[WebCore::legacyTIFFPasteboardType(), WebURLsWithTitlesPboardType, WebCore::legacyURLPasteboardType(), WebURLPboardType, WebURLNamePboardType, WebCore::legacyStringPasteboardType()];
    return types.get().get();
}

NSArray* PasteboardTypes::forImagesWithArchive()
{
    static NeverDestroyed<RetainPtr<NSArray>> types = @[WebCore::legacyTIFFPasteboardType(), WebURLsWithTitlesPboardType, WebCore::legacyURLPasteboardType(), WebURLPboardType, WebURLNamePboardType, WebCore::legacyStringPasteboardType(), WebCore::legacyRTFDPasteboardType(), WebArchivePboardType];
    return types.get().get();
}

NSArray* PasteboardTypes::forSelection()
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN 
    static NeverDestroyed<RetainPtr<NSArray>> types = @[WebArchivePboardType, (__bridge NSString *)kUTTypeWebArchive, WebCore::legacyRTFDPasteboardType(), WebCore::legacyRTFPasteboardType(), WebCore::legacyStringPasteboardType()];
ALLOW_DEPRECATED_DECLARATIONS_END
    return types.get().get();
}
    
} // namespace WebKit

#endif // PLATFORM(MAC)
