/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 18, 2021.
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
#import <WebKit/WKFoundation.h>

NS_ASSUME_NONNULL_BEGIN

@class _WKFrameHandle;

typedef NS_ENUM(NSInteger, _WKResourceLoadInfoResourceType) {
    _WKResourceLoadInfoResourceTypeApplicationManifest,
    _WKResourceLoadInfoResourceTypeBeacon,
    _WKResourceLoadInfoResourceTypeCSPReport,
    _WKResourceLoadInfoResourceTypeDocument,
    _WKResourceLoadInfoResourceTypeImage,
    _WKResourceLoadInfoResourceTypeFetch,
    _WKResourceLoadInfoResourceTypeFont,
    _WKResourceLoadInfoResourceTypeMedia,
    _WKResourceLoadInfoResourceTypeObject,
    _WKResourceLoadInfoResourceTypePing,
    _WKResourceLoadInfoResourceTypeScript,
    _WKResourceLoadInfoResourceTypeStylesheet,
    _WKResourceLoadInfoResourceTypeXMLHTTPRequest,
    _WKResourceLoadInfoResourceTypeXSLT,
    _WKResourceLoadInfoResourceTypeOther = -1,
};

WK_CLASS_AVAILABLE(macos(11.0), ios(14.0))
@interface _WKResourceLoadInfo : NSObject <NSSecureCoding>

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@property (nonatomic, readonly) uint64_t resourceLoadID;
@property (nonatomic, readonly) _WKFrameHandle *frame;
@property (nonatomic, readonly, nullable) _WKFrameHandle *parentFrame;
@property (nonatomic, readonly, nullable) NSUUID *documentID;
@property (nonatomic, readonly) NSURL *originalURL;
@property (nonatomic, readonly) NSString *originalHTTPMethod;
@property (nonatomic, readonly) NSDate *eventTimestamp;
@property (nonatomic, readonly) BOOL loadedFromCache;
@property (nonatomic, readonly) _WKResourceLoadInfoResourceType resourceType;

@end

NS_ASSUME_NONNULL_END
