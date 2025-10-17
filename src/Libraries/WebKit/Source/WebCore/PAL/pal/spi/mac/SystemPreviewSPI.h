/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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
#if USE(APPLE_INTERNAL_SDK)

#if ENABLE(ARKIT_INLINE_PREVIEW_MAC)
#import <AssetViewer/ASVInlinePreview.h>
#endif

#else // USE(APPLE_INTERNAL_SDK)

#if ENABLE(ARKIT_INLINE_PREVIEW_MAC)

#import <simd/simd.h>

NS_ASSUME_NONNULL_BEGIN

@class ASVInlinePreview;

@interface ASVInlinePreview : NSObject
@property (nonatomic, readonly) NSUUID *uuid;
@property (nonatomic, readonly) CALayer *layer;
@property (nonatomic, readonly) uint32_t contextId;

- (instancetype)initWithFrame:(CGRect)frame;
- (instancetype)initWithFrame:(CGRect)frame UUID:(NSUUID *)uuid;
- (void)setupRemoteConnectionWithCompletionHandler:(void (^)(NSError * _Nullable error))handler;
- (void)preparePreviewOfFileAtURL:(NSURL *)url completionHandler:(void (^)(NSError * _Nullable error))handler;
- (void)setRemoteContext:(uint32_t)contextId;

- (void)updateFrame:(CGRect)newFrame completionHandler:(void (^)(CAFenceHandle * _Nullable fenceHandle, NSError * _Nullable error))handler;
- (void)setFrameWithinFencedTransaction:(CGRect)frame;

- (void)mouseDownAtLocation:(CGPoint)location timestamp:(NSTimeInterval)timestamp;
- (void)mouseDraggedAtLocation:(CGPoint)location timestamp:(NSTimeInterval)timestamp;
- (void)mouseUpAtLocation:(CGPoint)location timestamp:(NSTimeInterval)timestamp;

typedef void (^ASVCameraTransformReplyBlock) (simd_float3 cameraTransform, NSError * _Nullable error);
- (void)getCameraTransform:(ASVCameraTransformReplyBlock)reply;
- (void)setCameraTransform:(simd_float3)transform;

@property (nonatomic, readwrite) NSTimeInterval currentTime;
@property (nonatomic, readonly) NSTimeInterval duration;
@property (nonatomic, readwrite) BOOL isLooping;
@property (nonatomic, readonly) BOOL isPlaying;
typedef void (^ASVSetIsPlayingReplyBlock) (BOOL isPlaying, NSError * _Nullable error);
- (void)setIsPlaying:(BOOL)isPlaying reply:(ASVSetIsPlayingReplyBlock)reply;

@property (nonatomic, readonly) BOOL hasAudio;
@property (nonatomic, readwrite) BOOL isMuted;

@end

NS_ASSUME_NONNULL_END

#endif // ENABLE(ARKIT_INLINE_PREVIEW_MAC)

#endif // USE(APPLE_INTERNAL_SDK)
