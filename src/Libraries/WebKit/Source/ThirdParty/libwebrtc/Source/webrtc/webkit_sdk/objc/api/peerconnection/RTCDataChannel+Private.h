/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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
#import "RTCDataChannel.h"

#include "api/data_channel_interface.h"
#include "api/scoped_refptr.h"

NS_ASSUME_NONNULL_BEGIN

@class RTCPeerConnectionFactory;

@interface RTCDataBuffer ()

/**
 * The native DataBuffer representation of this RTCDatabuffer object. This is
 * needed to pass to the underlying C++ APIs.
 */
@property(nonatomic, readonly) const webrtc::DataBuffer *nativeDataBuffer;

/** Initialize an RTCDataBuffer from a native DataBuffer. */
- (instancetype)initWithNativeBuffer:(const webrtc::DataBuffer &)nativeBuffer;

@end

@interface RTCDataChannel ()

/** Initialize an RTCDataChannel from a native DataChannelInterface. */
- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
              nativeDataChannel:(rtc::scoped_refptr<webrtc::DataChannelInterface>)nativeDataChannel
    NS_DESIGNATED_INITIALIZER;

+ (webrtc::DataChannelInterface::DataState)nativeDataChannelStateForState:
        (RTCDataChannelState)state;

+ (RTCDataChannelState)dataChannelStateForNativeState:
        (webrtc::DataChannelInterface::DataState)nativeState;

+ (NSString *)stringForState:(RTCDataChannelState)state;

@end

NS_ASSUME_NONNULL_END
