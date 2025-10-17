/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 27, 2022.
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

#if HAVE(ARKIT_QUICK_LOOK_PREVIEW_ITEM)
#import <AssetViewer/ARQuickLookWebKitItem.h>
#endif

#if PLATFORM(IOS) || PLATFORM(VISION)
#import <AssetViewer/ASVThumbnailView.h>
#endif

#if ENABLE(ARKIT_INLINE_PREVIEW_IOS)
#import <AssetViewer/ASVInlinePreview.h>
#endif

#if PLATFORM(VISION)
#import <AssetViewer/ASVLaunchPreview.h>
#endif

#else

#import <UIKit/UIKit.h>

#if PLATFORM(IOS) || PLATFORM(VISION)
#import <pal/spi/ios/QuickLookSPI.h>

@class ASVThumbnailView;
@class QLPreviewController;

NS_ASSUME_NONNULL_BEGIN

@protocol ASVThumbnailViewDelegate <NSObject>
- (void)thumbnailView:(ASVThumbnailView *)thumbnailView wantsToPresentPreviewController:(QLPreviewController *)previewController forItem:(QLItem *)item;
@end

@interface ASVThumbnailView : UIView
@property (nonatomic, weak) id<ASVThumbnailViewDelegate> delegate;
@property (nonatomic, assign) QLItem *thumbnailItem;
@property (nonatomic) CGSize maxThumbnailSize;
@end

NS_ASSUME_NONNULL_END

#endif

#if HAVE(ARKIT_QUICK_LOOK_PREVIEW_ITEM)
#import <ARKit/ARKit.h>

NS_ASSUME_NONNULL_BEGIN

#if PLATFORM(VISION)
@interface ARQuickLookPreviewItem : NSObject
@property (nonatomic, strong, nullable) NSURL *canonicalWebPageURL;
- (instancetype)initWithFileAtURL:(NSURL *)url;
@end
#endif

@protocol ARQuickLookWebKitItemDelegate
@end

@class ARQuickLookWebKitItem;

@interface ARQuickLookWebKitItem : QLItem
- (instancetype)initWithPreviewItemProvider:(NSItemProvider *)itemProvider contentType:(NSString *)contentType previewTitle:(NSString *)previewTitle fileSize:(NSNumber *)fileSize previewItem:(ARQuickLookPreviewItem *)previewItem;
- (void)setDelegate:(id <ARQuickLookWebKitItemDelegate>)delegate;
@end

NS_ASSUME_NONNULL_END

#endif

#if ENABLE(ARKIT_INLINE_PREVIEW_IOS)

#import <simd/simd.h>

NS_ASSUME_NONNULL_BEGIN

@class ASVInlinePreview;
@class CAFenceHandle;
@interface ASVInlinePreview : NSObject
@property (nonatomic, readonly) CALayer *layer;

- (instancetype)initWithFrame:(CGRect)frame;
- (void)setupRemoteConnectionWithCompletionHandler:(void (^)(NSError * _Nullable error))handler;
- (void)preparePreviewOfFileAtURL:(NSURL *)url completionHandler:(void (^)(NSError * _Nullable error))handler;
- (void)updateFrame:(CGRect)newFrame completionHandler:(void (^)(CAFenceHandle * _Nullable fenceHandle, NSError * _Nullable error))handler;
- (void)setFrameWithinFencedTransaction:(CGRect)frame;
- (void)createFullscreenInstanceWithInitialFrame:(CGRect)initialFrame previewOptions:(NSDictionary *)previewOptions completionHandler:(void (^)(UIViewController *remoteViewController, CAFenceHandle * _Nullable fenceHandle, NSError * _Nullable error))handler;
- (void)observeDismissFullscreenWithCompletionHandler:(void (^)(CAFenceHandle * _Nullable fenceHandle, NSDictionary * _Nonnull payload, NSError * _Nullable error))handler;
- (void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(nullable UIEvent *)event;
- (void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(nullable UIEvent *)event;
- (void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(nullable UIEvent *)event;
- (void)touchesCancelled:(NSSet<UITouch *> *)touches withEvent:(nullable UIEvent *)event;

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

@property (nonatomic, retain, nullable) NSURL *canonicalWebPageURL;
@property (nonatomic, retain, nullable) NSString *urlFragment;

@end

@interface ASVLaunchPreview : NSObject
+ (void)beginPreviewApplicationWithURLs:(NSArray *)urls is3DContent:(BOOL)is3DContent websiteURL:(NSURL *)websiteURL completion:(void (^)(NSError *))handler;
+ (void)launchPreviewApplicationWithURLs:(NSArray *)urls completion:(void (^)(NSError *))handler;
+ (void)cancelPreviewApplicationWithURLs:(NSArray *)urls error:(NSError * _Nullable)error completion:(void (^)(NSError *))handler;
@end

NS_ASSUME_NONNULL_END

#endif

#endif
