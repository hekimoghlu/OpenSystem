/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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
#import <Foundation/Foundation.h>
#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#else
#import <AppKit/AppKit.h>
#endif

#import "base/RTCVideoFrame.h"

NS_ASSUME_NONNULL_BEGIN
/**
 * Protocol defining ability to render RTCVideoFrame in Metal enabled views.
 */
@protocol RTCMTLRenderer <NSObject>

/**
 * Method to be implemented to perform actual rendering of the provided frame.
 *
 * @param frame The frame to be rendered.
 */
- (void)drawFrame:(RTCVideoFrame *)frame;

/**
 * Sets the provided view as rendering destination if possible.
 *
 * If not possible method returns NO and callers of the method are responisble for performing
 * cleanups.
 */

#if TARGET_OS_IOS
- (BOOL)addRenderingDestination:(__kindof UIView *)view;
#else
- (BOOL)addRenderingDestination:(__kindof NSView *)view;
#endif

@end

/**
 * Implementation of RTCMTLRenderer protocol.
 */
NS_AVAILABLE(10_11, 9_0)
@interface RTCMTLRenderer : NSObject <RTCMTLRenderer>

/** @abstract   A wrapped RTCVideoRotation, or nil.
    @discussion When not nil, the rotation of the actual frame is ignored when rendering.
 */
@property(atomic, nullable) NSValue *rotationOverride;

@end

NS_ASSUME_NONNULL_END
