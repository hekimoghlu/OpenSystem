/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 30, 2022.
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

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, WKSLinearMediaContentMode) {
    WKSLinearMediaContentModeNone = 0,
    WKSLinearMediaContentModeScaleAspectFit,
    WKSLinearMediaContentModeScaleAspectFill,
    WKSLinearMediaContentModeScaleToFill
};

typedef NS_ENUM(NSInteger, WKSLinearMediaContentType) {
    WKSLinearMediaContentTypeNone = 0,
    WKSLinearMediaContentTypeImmersive,
    WKSLinearMediaContentTypeSpatial,
    WKSLinearMediaContentTypePlanar,
    WKSLinearMediaContentTypeAudioOnly
};

typedef NS_ENUM(NSInteger, WKSLinearMediaPresentationState) {
    WKSLinearMediaPresentationStateInline = 0,
    WKSLinearMediaPresentationStateEnteringFullscreen,
    WKSLinearMediaPresentationStateFullscreen,
    WKSLinearMediaPresentationStateExitingFullscreen
};

typedef NS_ENUM(NSInteger, WKSLinearMediaViewingMode) {
    WKSLinearMediaViewingModeNone = 0,
    WKSLinearMediaViewingModeMono,
    WKSLinearMediaViewingModeStereo,
    WKSLinearMediaViewingModeImmersive,
    WKSLinearMediaViewingModeSpatial
};

API_AVAILABLE(visionos(1.0))
@interface WKSLinearMediaContentMetadata : NSObject
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithTitle:(nullable NSString *)title subtitle:(nullable NSString *)subtitle NS_DESIGNATED_INITIALIZER;
@property (nonatomic, readonly, copy, nullable) NSString *title;
@property (nonatomic, readonly, copy, nullable) NSString *subtitle;
@end

API_AVAILABLE(visionos(1.0))
@interface WKSLinearMediaTimeRange : NSObject
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithLowerBound:(NSTimeInterval)lowerBound upperBound:(NSTimeInterval)upperBound NS_DESIGNATED_INITIALIZER;
@property (nonatomic, readonly) NSTimeInterval lowerBound;
@property (nonatomic, readonly) NSTimeInterval upperBound;
@end

API_AVAILABLE(visionos(1.0))
@interface WKSLinearMediaTrack : NSObject
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithLocalizedDisplayName:(NSString *)localizedDisplayName NS_DESIGNATED_INITIALIZER;
@property (nonatomic, readonly, copy) NSString *localizedDisplayName;
@end

API_AVAILABLE(visionos(1.0))
@interface WKSLinearMediaSpatialVideoMetadata : NSObject
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithWidth:(SInt32)width height:(SInt32)height horizontalFOVDegrees:(float)horizontalFOVDegrees baseline:(float)baseline disparityAdjustment:(float)disparityAdjustment NS_DESIGNATED_INITIALIZER;
@property (nonatomic, readonly) SInt32 width;
@property (nonatomic, readonly) SInt32 height;
@property (nonatomic, readonly) float horizontalFOVDegrees;
@property (nonatomic, readonly) float baseline;
@property (nonatomic, readonly) float disparityAdjustment;
@end

NS_ASSUME_NONNULL_END

#endif /* defined(TARGET_OS_VISION) && TARGET_OS_VISION */
