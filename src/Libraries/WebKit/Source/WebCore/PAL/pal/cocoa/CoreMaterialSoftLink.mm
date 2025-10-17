/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
#include "config.h"

#if HAVE(CORE_MATERIAL)

#import <pal/spi/cocoa/CoreMaterialSPI.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_PRIVATE_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(PAL, CoreMaterial, PAL_EXPORT)

SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, CoreMaterial, MTMaterialLayer, PAL_EXPORT)

SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, CoreMaterial, MTCoreMaterialRecipePlatformContentLight, NSString *, PAL_EXPORT)
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, CoreMaterial, MTCoreMaterialRecipePlatformChromeLight, NSString *, PAL_EXPORT)
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, CoreMaterial, MTCoreMaterialRecipePlatformContentThickLight, NSString *, PAL_EXPORT)
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, CoreMaterial, MTCoreMaterialRecipePlatformContentThinLight, NSString *, PAL_EXPORT)
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, CoreMaterial, MTCoreMaterialRecipePlatformContentUltraThinLight, NSString *, PAL_EXPORT)

#endif // HAVE(CORE_MATERIAL)
