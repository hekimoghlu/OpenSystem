/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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

#if ((USE(SYSTEM_PREVIEW) && HAVE(ARKIT_QUICK_LOOK_PREVIEW_ITEM)) || ((PLATFORM(IOS) || PLATFORM(VISION)) && USE(APPLE_INTERNAL_SDK) && __has_include(<WebKitAdditions/ARKitSoftLinkAdditions.mm>)))

#import <simd/simd.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_SOURCE(WebKit, ARKit);

SOFT_LINK_CLASS_FOR_SOURCE(WebKit, ARKit, ARQuickLookPreviewItem);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, ARKit, ARSession);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, ARKit, ARWorldTrackingConfiguration);

SOFT_LINK_FUNCTION_FOR_SOURCE(WebKit, ARKit, ARMatrixMakeLookAt, simd_float4x4, (simd_float3 origin, simd_float3 direction), (origin, direction));

#if (PLATFORM(IOS) || PLATFORM(VISION)) && USE(APPLE_INTERNAL_SDK) && __has_include(<WebKitAdditions/ARKitSoftLinkAdditions.mm>)
#import <WebKitAdditions/ARKitSoftLinkAdditions.mm>
#endif

#endif
