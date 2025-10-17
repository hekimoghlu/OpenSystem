/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#import <AVFoundation/AVAssetWriter_Private.h>

#else

NS_ASSUME_NONNULL_BEGIN

@interface AVFragmentedMediaDataReport : NSObject
@end

#if !HAVE(AVASSETWRITERDELEGATE_API)
@protocol AVAssetWriterDelegate <NSObject>
@optional
- (void)assetWriter:(AVAssetWriter *)assetWriter didProduceFragmentedHeaderData:(NSData *)fragmentedHeaderData;
- (void)assetWriter:(AVAssetWriter *)assetWriter didProduceFragmentedMediaData:(NSData *)fragmentedMediaData fragmentedMediaDataReport:(AVFragmentedMediaDataReport *)fragmentedMediaDataReport;
@end
#endif

@interface AVAssetWriter ()
- (nullable instancetype)initWithFileType:(NSString * _Nullable)outputFileType error:(NSError * _Nullable * _Nullable)outError;
- (void)flush;
#if !HAVE(AVASSETWRITERDELEGATE_API)
@property (weak, nullable) id <AVAssetWriterDelegate> delegate SPI_AVAILABLE(macos(10.15), ios(13.0), tvos(13.0), watchos(6.0));
#endif
@end

NS_ASSUME_NONNULL_END

#endif // USE(APPLE_INTERNAL_SDK)
