/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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

#include <ImageIO/ImageIOBase.h> 

#if USE(APPLE_INTERNAL_SDK)
#include <ImageIO/CGImageSourcePrivate.h>
#endif

IMAGEIO_EXTERN const CFStringRef kCGImageSourceShouldPreferRGB32;
IMAGEIO_EXTERN const CFStringRef kCGImageSourceSkipMetadata;
IMAGEIO_EXTERN const CFStringRef kCGImageSourceSubsampleFactor;
IMAGEIO_EXTERN const CFStringRef kCGImageSourceShouldCacheImmediately;
IMAGEIO_EXTERN const CFStringRef kCGImageSourceUseHardwareAcceleration;

WTF_EXTERN_C_BEGIN
CFStringRef CGImageSourceGetTypeWithData(CFDataRef, CFStringRef, bool*);
#if HAVE(CGIMAGESOURCE_WITH_SET_ALLOWABLE_TYPES)
OSStatus CGImageSourceSetAllowableTypes(CFArrayRef allowableTypes);
#endif

#if HAVE(CGIMAGESOURCE_DISABLE_HARDWARE_DECODING)
IMAGEIO_EXTERN OSStatus CGImageSourceDisableHardwareDecoding();
#endif

#if HAVE(CGIMAGESOURCE_ENABLE_RESTRICTED_DECODING)
IMAGEIO_EXTERN OSStatus CGImageSourceEnableRestrictedDecoding();
#endif

WTF_EXTERN_C_END
