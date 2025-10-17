/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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

#if USE(MEDIATOOLBOX)

#include <MediaToolbox/MediaToolbox.h>

#include <pal/spi/cf/CoreMediaSPI.h>
#include <wtf/spi/darwin/XPCSPI.h>

WTF_EXTERN_C_BEGIN
typedef struct OpaqueFigVideoTarget *FigVideoTargetRef;
OSStatus FigVideoTargetCreateWithVideoReceiverEndpointID(CFAllocatorRef, xpc_object_t videoReceiverXPCEndpointID, CFDictionaryRef creationOptions, FigVideoTargetRef* videoTargetOut);
WTF_EXTERN_C_END

// FIXME (68673547): Use actual <MediaToolbox/FigPhoto.h> and FigPhotoContainerFormat enum when we weak-link instead of soft-link MediaToolbox and CoreMedia.
#define kPALFigPhotoContainerFormat_HEIF 0
#define kPALFigPhotoContainerFormat_JFIF 1

#endif // USE(MEDIATOOLBOX)
