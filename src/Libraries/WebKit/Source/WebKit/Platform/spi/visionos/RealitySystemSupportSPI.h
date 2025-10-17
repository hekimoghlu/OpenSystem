/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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

#if USE(APPLE_INTERNAL_SDK)
#import <RealitySystemSupport/RealitySystemSupport.h>
#else

#import <QuartzCore/CALayer.h>

@class CARemoteEffectGroup;

typedef NS_OPTIONS(NSUInteger, RCPRemoteEffectInputTypes) {
    RCPRemoteEffectInputTypeGaze = 1 << 0,
    RCPRemoteEffectInputTypeDirectTouch = 1 << 1,
    RCPRemoteEffectInputTypePointer = 1 << 2,
    RCPRemoteEffectInputTypeNearbyTouchLeftHand = 1 << 3,
    RCPRemoteEffectInputTypeNearbyTouchRightHand = 1 << 4,
    RCPRemoteEffectInputTypeNearbyTouchAnyHand = (RCPRemoteEffectInputTypeNearbyTouchLeftHand | RCPRemoteEffectInputTypeNearbyTouchRightHand),
    RCPRemoteEffectInputTypeAnyTouch = (RCPRemoteEffectInputTypeDirectTouch | RCPRemoteEffectInputTypeNearbyTouchAnyHand),
    RCPRemoteEffectInputTypeIndirect = RCPRemoteEffectInputTypeGaze,
    RCPRemoteEffectInputTypesNone = 0,
    RCPRemoteEffectInputTypesAll = ~0ul
};

typedef NS_OPTIONS(NSUInteger, RCPGlowEffectContentRenderingHints) {
    RCPGlowEffectContentRenderingHintPhoto = 1 << 0,
    RCPGlowEffectContentRenderingHintHasMotion = 1 << 1,
};

@interface RCPGlowEffectLayer : CALayer
@property (nonatomic, copy) void (^effectGroupConfigurator)(CARemoteEffectGroup *group);
@property (nonatomic) RCPGlowEffectContentRenderingHints contentRenderingHints;

- (void)setBrightnessMultiplier:(CGFloat)multiplier forInputTypes:(RCPRemoteEffectInputTypes)inputTypes;
@end

#endif // USE(APPLE_INTERNAL_SDK)
