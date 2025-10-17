/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
#import <QuickLook/QuickLook.h>

#if USE(APPLE_INTERNAL_SDK)

#import <QuickLook/QuickLookPrivate.h>

#else

@interface QLPreviewConverter : NSObject
@end

@interface QLPreviewConverter ()
- (NSURLRequest *)safeRequestForRequest:(NSURLRequest *)request;
- (id)initWithConnection:(NSURLConnection *)connection delegate:(id)delegate response:(NSURLResponse *)response options:(NSDictionary *)options;
- (id)initWithData:(NSData *)data name:(NSString *)name uti:(NSString *)uti options:(NSDictionary *)options;
- (void)appendData:(NSData *)data;
- (void)appendDataArray:(NSArray *)dataArray;
- (void)finishConverting;
- (void)finishedAppendingData;
@property (readonly, nonatomic) NSString *previewFileName;
@property (readonly, nonatomic) NSString *previewUTI;
@property (readonly, nonatomic) NSURLRequest *previewRequest;
@property (readonly, nonatomic) NSURLResponse *previewResponse;
@end

@class QLItem;

@protocol QLPreviewItemDataProvider <NSObject>
- (NSData *)provideDataForItem:(QLItem *)item;
@end

@interface QLItem : NSObject<QLPreviewItem>
- (instancetype)initWithDataProvider:(id<QLPreviewItemDataProvider>)data contentType:(NSString *)contentType previewTitle:(NSString *)previewTitle;
- (instancetype)initWithPreviewItemProvider:(NSItemProvider *)itemProvider contentType:(NSString *)contentType previewTitle:(NSString *)previewTitle fileSize:(NSNumber *)fileSize;
- (void)setPreviewItemProviderProgress:(NSNumber*)progress;
- (void)setUseLoadingTimeout:(BOOL) timeout;
@property (nonatomic, copy) NSDictionary *previewOptions;
@end

#define kQLReturnPasswordProtected 1 << 2

typedef NS_OPTIONS(NSUInteger, QLPreviewControllerFirstTimeAppearanceActions) {
    QLPreviewControllerFirstTimeAppearanceActionNone = 0,
    QLPreviewControllerFirstTimeAppearanceActionPlayAudio = 1 << 0,
    QLPreviewControllerFirstTimeAppearanceActionPlayVideo = 1 << 1,
    QLPreviewControllerFirstTimeAppearanceActionEnableEditMode = 1 << 2,
    QLPreviewControllerFirstTimeAppearanceActionEnableVisualSearchDataDetection = 1 << 3,
    QLPreviewControllerFirstTimeAppearanceActionEnableVisualSearchMode = 1 << 4,
    QLPreviewControllerFirstTimeAppearanceActionAll = NSUIntegerMax,
};

@interface QLPreviewController ()
@property (nonatomic, assign) QLPreviewControllerFirstTimeAppearanceActions appearanceActions;
@end

#endif

static_assert(kQLReturnPasswordProtected == 4, "kQLReturnPasswordProtected should equal 4.");

WTF_EXTERN_C_BEGIN

NSSet *QLPreviewGetSupportedMIMETypes();
NSString *QLTypeCopyBestMimeTypeForFileNameAndMimeType(NSString *fileName, NSString *mimeType);
NSString *QLTypeCopyBestMimeTypeForURLAndMimeType(NSURL *, NSString *mimeType);
NSString *QLTypeCopyUTIForURLAndMimeType(NSURL *, NSString *mimeType);

WTF_EXTERN_C_END
