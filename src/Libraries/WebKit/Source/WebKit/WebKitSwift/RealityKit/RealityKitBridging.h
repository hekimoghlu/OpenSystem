/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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

#if defined(TARGET_OS_VISION) && TARGET_OS_VISION

#import <simd/simd.h>

typedef struct REEntity *REEntityRef;

NS_ASSUME_NONNULL_BEGIN

typedef struct {
    simd_float3 scale;
    simd_quatf rotation;
    simd_float3 translation;
} WKEntityTransform;

@protocol WKSRKEntityDelegate <NSObject>
@optional
- (void)entityAnimationPlaybackStateDidUpdate:(id)entity;
@end

@interface WKSRKEntity : NSObject
@property (nonatomic, weak) id <WKSRKEntityDelegate> delegate;
@property (nonatomic, copy, nullable) NSString *name;

@property (nonatomic, readonly) simd_float3 boundingBoxExtents;
@property (nonatomic, readonly) simd_float3 boundingBoxCenter;
@property (nonatomic) WKEntityTransform transform;
@property (nonatomic) float opacity;
@property (nonatomic, readonly) NSTimeInterval duration;
@property (nonatomic) bool loop;
@property (nonatomic) float playbackRate;
@property (nonatomic) bool paused;
@property (nonatomic) NSTimeInterval currentTime;

+ (bool)isLoadFromDataAvailable;
+ (void)loadFromData:(NSData *)data completionHandler:(void (^)(WKSRKEntity * _Nullable entity))completionHandler;
- (instancetype)initWithCoreEntity:(REEntityRef)coreEntity;
- (void)setParentCoreEntity:(REEntityRef)parentCoreEntity;
- (void)setUpAnimationWithAutoPlay:(BOOL)autoPlay;
- (void)applyIBLData:(NSData *)data withCompletion:(void (^)(BOOL success))completion;
- (void)removeIBL;
@end

NS_ASSUME_NONNULL_END

#endif // defined(TARGET_OS_VISION) && TARGET_OS_VISION
