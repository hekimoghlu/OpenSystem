/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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
#pragma once

#if USE(QUICK_LOOK)

#include <pal/spi/ios/QuickLookSPI.h>
#include <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(PAL, QuickLook)

SOFT_LINK_CLASS_FOR_HEADER(PAL, QLItem)
SOFT_LINK_CLASS_FOR_HEADER(PAL, QLPreviewController)
SOFT_LINK_CLASS_FOR_HEADER(PAL, QLPreviewConverter)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, QuickLook, kQLPreviewOptionPasswordKey, CFStringRef);
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, QuickLook, QLPreviewGetSupportedMIMETypes, NSSet *, (), ())
#define QLPreviewGetSupportedMIMETypes softLink_QuickLook_QLPreviewGetSupportedMIMETypes
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, QuickLook, QLTypeCopyBestMimeTypeForFileNameAndMimeType, NSString *, (NSString *fileName, NSString *mimeType), (fileName, mimeType))
#define QLTypeCopyBestMimeTypeForFileNameAndMimeType softLink_QuickLook_QLTypeCopyBestMimeTypeForFileNameAndMimeType
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, QuickLook, QLTypeCopyBestMimeTypeForURLAndMimeType, NSString *, (NSURL *url, NSString *mimeType), (url, mimeType))
#define QLTypeCopyBestMimeTypeForURLAndMimeType softLink_QuickLook_QLTypeCopyBestMimeTypeForURLAndMimeType
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, QuickLook, QLTypeCopyUTIForURLAndMimeType, NSString *, (NSURL *url, NSString *mimeType), (url, mimeType))
#define QLTypeCopyUTIForURLAndMimeType softLink_QuickLook_QLTypeCopyUTIForURLAndMimeType

#endif // USE(QUICK_LOOK)
