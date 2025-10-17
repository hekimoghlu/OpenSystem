/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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

#if HAVE(VK_IMAGE_ANALYSIS)

#import <pal/spi/cocoa/VisionKitCoreSPI.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_PRIVATE_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(PAL, VisionKitCore, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT_AND_IS_OPTIONAL(PAL, VisionKitCore, VKImageAnalyzer, PAL_EXPORT, true)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT_AND_IS_OPTIONAL(PAL, VisionKitCore, VKImageAnalyzerRequest, PAL_EXPORT, true)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT_AND_IS_OPTIONAL(PAL, VisionKitCore, VKCImageAnalyzer, PAL_EXPORT, true)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT_AND_IS_OPTIONAL(PAL, VisionKitCore, VKCImageAnalyzerRequest, PAL_EXPORT, true)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT_AND_IS_OPTIONAL(PAL, VisionKitCore, VKCImageAnalysis, PAL_EXPORT, true)
#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT_AND_IS_OPTIONAL(PAL, VisionKitCore, VKCImageAnalysisInteraction, PAL_EXPORT, true)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT_AND_IS_OPTIONAL(PAL, VisionKitCore, VKCImageAnalysisOverlayView, PAL_EXPORT, true)
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, VisionKitCore, vk_cgImageRemoveBackground, void, (CGImageRef image, BOOL crop, VKCGImageRemoveBackgroundCompletion completion), (image, crop, completion), PAL_EXPORT)
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, VisionKitCore, vk_cgImageRemoveBackgroundWithDownsizing, void, (CGImageRef image, BOOL canDownsize, BOOL cropToFit, void(^completion)(CGImageRef, NSError *)), (image, canDownsize, cropToFit, completion), PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT_AND_IS_OPTIONAL(PAL, VisionKitCore, VKCRemoveBackgroundRequestHandler, PAL_EXPORT, true)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT_AND_IS_OPTIONAL(PAL, VisionKitCore, VKCRemoveBackgroundRequest, PAL_EXPORT, true)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT_AND_IS_OPTIONAL(PAL, VisionKitCore, VKCRemoveBackgroundResult, PAL_EXPORT, true)
#endif

#endif // HAVE(VK_IMAGE_ANALYSIS)
