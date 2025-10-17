/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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

#if USE(QUICK_LOOK)

#import <pal/spi/ios/QuickLookSPI.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_SOURCE(PAL, QuickLook)

SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, QuickLook, QLItem, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, QuickLook, QLPreviewController, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, QuickLook, QLPreviewConverter)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, QuickLook, kQLPreviewOptionPasswordKey, CFStringRef);
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, QuickLook, QLPreviewGetSupportedMIMETypes, NSSet *, (), ())
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, QuickLook, QLTypeCopyBestMimeTypeForFileNameAndMimeType, NSString *, (NSString *fileName, NSString *mimeType), (fileName, mimeType))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, QuickLook, QLTypeCopyBestMimeTypeForURLAndMimeType, NSString *, (NSURL *url, NSString *mimeType), (url, mimeType))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, QuickLook, QLTypeCopyUTIForURLAndMimeType, NSString *, (NSURL *url, NSString *mimeType), (url, mimeType))

#endif // USE(QUICK_LOOK)
