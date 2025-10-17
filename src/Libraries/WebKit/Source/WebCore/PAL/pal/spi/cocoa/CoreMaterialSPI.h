/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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

#if HAVE(CORE_MATERIAL)

#import <Foundation/NSString.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>

#if USE(APPLE_INTERNAL_SDK)

#import <CoreMaterial/CoreMaterial.h>

#else

typedef NSString * MTCoreMaterialRecipe NS_STRING_ENUM;

extern MTCoreMaterialRecipe const MTCoreMaterialRecipePlatformChromeLight;
extern MTCoreMaterialRecipe const MTCoreMaterialRecipePlatformContentUltraThinLight;
extern MTCoreMaterialRecipe const MTCoreMaterialRecipePlatformContentThinLight;
extern MTCoreMaterialRecipe const MTCoreMaterialRecipePlatformContentLight;
extern MTCoreMaterialRecipe const MTCoreMaterialRecipePlatformContentThickLight;

@interface MTMaterialLayer : CABackdropLayer

@property (nonatomic, copy) MTCoreMaterialRecipe recipe;

@end

#endif // USE(APPLE_INTERNAL_SDK)

#endif // HAVE(CORE_MATERIAL)
