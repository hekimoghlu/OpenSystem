/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 2, 2022.
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

#include <CoreMedia/CoreMedia.h>

NS_ASSUME_NONNULL_BEGIN

@protocol AVStreamDataParserOutputHandling;
@class AVStreamDataParserInternal;

@interface AVStreamDataParser : NSObject {
@private
    AVStreamDataParserInternal *_parser;
}

- (void)setDelegate:(nullable id<AVStreamDataParserOutputHandling>)delegate;
- (void)appendStreamData:(NSData *)data;
typedef NS_ENUM(NSUInteger, AVStreamDataParserStreamDataFlags) {
    AVStreamDataParserStreamDataDiscontinuity = 1 << 0,
};
- (void)appendStreamData:(NSData *)data withFlags:(AVStreamDataParserStreamDataFlags)flags;
- (void)providePendingMediaData;
- (void)setShouldProvideMediaData:(BOOL)shouldProvideMediaData forTrackID:(CMPersistentTrackID)trackID;
- (BOOL)shouldProvideMediaDataForTrackID:(CMPersistentTrackID)trackID;
@end

@protocol AVStreamDataParserOutputHandling <NSObject>
typedef NS_ENUM(NSUInteger, AVStreamDataParserOutputMediaDataFlags) {
    AVStreamDataParserOutputMediaDataReserved = 1 << 0
};
@optional
- (void)streamDataParser:(AVStreamDataParser *)streamDataParser didParseStreamDataAsAsset:(AVAsset *)asset withDiscontinuity:(BOOL)withDiscontinuity;
- (void)streamDataParser:(AVStreamDataParser *)streamDataParser didFailToParseStreamDataWithError:(NSError *)err;
- (void)streamDataParser:(AVStreamDataParser *)streamDataParser didProvideMediaData:(CMSampleBufferRef)mediaData forTrackID:(CMPersistentTrackID)trackID mediaType:(AVMediaType)mediaType flags:(AVStreamDataParserOutputMediaDataFlags)flags;
- (void)streamDataParser:(AVStreamDataParser *)streamDataParser didReachEndOfTrackWithTrackID:(CMPersistentTrackID)trackID mediaType:(AVMediaType)mediaType;
- (void)streamDataParserWillProvideContentKeyRequestInitializationData:(AVStreamDataParser *)streamDataParser forTrackID:(CMPersistentTrackID)trackID;
- (void)streamDataParser:(AVStreamDataParser *)streamDataParser didProvideContentKeyRequestInitializationData:(NSData *)contentKeyRequestInitializationData forTrackID:(CMPersistentTrackID)trackID;

@end

@interface AVStreamDataParser (AVStreamDataParserTypeSupport)
+ (NSString *)outputMIMECodecParameterForInputMIMECodecParameter:(NSString *)inputMIMECodecParameter;

@end

@interface AVStreamDataParser (AVStreamDataParserContentKeyEligibility) <AVContentKeyRecipient>
@end

NS_ASSUME_NONNULL_END
