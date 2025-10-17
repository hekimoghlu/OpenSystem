/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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

#import "RTCMacros.h"

NS_ASSUME_NONNULL_BEGIN

RTC_OBJC_EXPORT
@interface RTCIceCandidate : NSObject

/**
 * If present, the identifier of the "media stream identification" for the media
 * component this candidate is associated with.
 */
@property(nonatomic, readonly, nullable) NSString *sdpMid;

/**
 * The index (starting at zero) of the media description this candidate is
 * associated with in the SDP.
 */
@property(nonatomic, readonly) int sdpMLineIndex;

/** The SDP string for this candidate. */
@property(nonatomic, readonly) NSString *sdp;

/** The URL of the ICE server which this candidate is gathered from. */
@property(nonatomic, readonly, nullable) NSString *serverUrl;

- (instancetype)init NS_UNAVAILABLE;

/**
 * Initialize an RTCIceCandidate from SDP.
 */
- (instancetype)initWithSdp:(NSString *)sdp
              sdpMLineIndex:(int)sdpMLineIndex
                     sdpMid:(nullable NSString *)sdpMid NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
