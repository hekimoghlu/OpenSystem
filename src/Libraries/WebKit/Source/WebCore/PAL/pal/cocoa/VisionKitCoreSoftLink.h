/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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

#if HAVE(VK_IMAGE_ANALYSIS)

#import <pal/spi/cocoa/VisionKitCoreSPI.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(PAL, VisionKitCore)
SOFT_LINK_CLASS_FOR_HEADER(PAL, VKImageAnalyzer)
SOFT_LINK_CLASS_FOR_HEADER(PAL, VKImageAnalyzerRequest)
SOFT_LINK_CLASS_FOR_HEADER(PAL, VKCImageAnalyzer)
SOFT_LINK_CLASS_FOR_HEADER(PAL, VKCImageAnalyzerRequest)
SOFT_LINK_CLASS_FOR_HEADER(PAL, VKCImageAnalysis)
#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)
SOFT_LINK_CLASS_FOR_HEADER(PAL, VKCImageAnalysisInteraction)
SOFT_LINK_CLASS_FOR_HEADER(PAL, VKCImageAnalysisOverlayView)
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_HEADER(PAL, VisionKitCore, vk_cgImageRemoveBackground, void, (CGImageRef image, BOOL crop, VKCGImageRemoveBackgroundCompletion completion), (image, crop, completion))
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_HEADER(PAL, VisionKitCore, vk_cgImageRemoveBackgroundWithDownsizing, void, (CGImageRef image, BOOL canDownsize, BOOL cropToFit, void(^completion)(CGImageRef, NSError *)), (image, canDownsize, cropToFit, completion))
SOFT_LINK_CLASS_FOR_HEADER(PAL, VKCRemoveBackgroundRequestHandler)
SOFT_LINK_CLASS_FOR_HEADER(PAL, VKCRemoveBackgroundRequest)
SOFT_LINK_CLASS_FOR_HEADER(PAL, VKCRemoveBackgroundResult)
#endif

#endif // HAVE(VK_IMAGE_ANALYSIS)
