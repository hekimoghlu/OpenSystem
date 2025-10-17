/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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

#if ENABLE(WEBXR) && USE(ARKITXR_IOS)

#import <Metal/Metal.h>
#import <UIKit/UIKit.h>
#import <WebCore/PlatformXR.h>

NS_ASSUME_NONNULL_BEGIN

@class ARFrame;
@class ARSession;

@interface WKARPresentationSessionDescriptor : NSObject <NSCopying>
@property (nonatomic, readwrite) MTLPixelFormat colorFormat;
@property (nonatomic, readwrite) MTLTextureUsage colorUsage;
@property (nonatomic, readwrite) NSUInteger rasterSampleCount;
@property (nonatomic, nullable, weak, readwrite) UIViewController *presentingViewController;
@end

@protocol WKARPresentationSession <NSObject>
@property (nonatomic, readonly) UIView *view;
@property (nonatomic, retain, readonly) ARFrame *currentFrame;
@property (nonatomic, retain, readonly) ARSession *session;
@property (nonatomic, nonnull, retain, readonly) id<MTLSharedEvent> completionEvent;
@property (nonatomic, nullable, retain, readonly) id<MTLTexture> colorTexture;
@property (nonatomic, readonly) NSUInteger renderingFrameIndex;
@property (atomic, readonly, getter=isSessionEndRequested) BOOL sessionEndRequested;

- (NSUInteger)startFrame;
- (Vector<PlatformXR::FrameData::InputSource>)collectInputSources;
- (void)present;
- (void)terminate;
@end

id<WKARPresentationSession> createPresentationSession(ARSession *, WKARPresentationSessionDescriptor *);

NS_ASSUME_NONNULL_END

#endif // ENABLE(WEBXR) && USE(ARKITXR_IOS)
