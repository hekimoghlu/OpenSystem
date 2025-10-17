/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
#import <AvailabilityMacros.h>
#import <Foundation/Foundation.h>

#import "RTCMacros.h"

NS_ASSUME_NONNULL_BEGIN

RTC_OBJC_EXPORT
@interface RTCDataBuffer : NSObject

/** NSData representation of the underlying buffer. */
@property(nonatomic, readonly) NSData *data;

/** Indicates whether |data| contains UTF-8 or binary data. */
@property(nonatomic, readonly) BOOL isBinary;

- (instancetype)init NS_UNAVAILABLE;

/**
 * Initialize an RTCDataBuffer from NSData. |isBinary| indicates whether |data|
 * contains UTF-8 or binary data.
 */
- (instancetype)initWithData:(NSData *)data isBinary:(BOOL)isBinary;

@end

@class RTCDataChannel;
RTC_OBJC_EXPORT
@protocol RTCDataChannelDelegate <NSObject>

/** The data channel state changed. */
- (void)dataChannelDidChangeState:(RTCDataChannel *)dataChannel;

/** The data channel successfully received a data buffer. */
- (void)dataChannel:(RTCDataChannel *)dataChannel
    didReceiveMessageWithBuffer:(RTCDataBuffer *)buffer;

@optional
/** The data channel's |bufferedAmount| changed. */
- (void)dataChannel:(RTCDataChannel *)dataChannel didChangeBufferedAmount:(uint64_t)amount;

@end

/** Represents the state of the data channel. */
typedef NS_ENUM(NSInteger, RTCDataChannelState) {
  RTCDataChannelStateConnecting,
  RTCDataChannelStateOpen,
  RTCDataChannelStateClosing,
  RTCDataChannelStateClosed,
};

RTC_OBJC_EXPORT
@interface RTCDataChannel : NSObject

/**
 * A label that can be used to distinguish this data channel from other data
 * channel objects.
 */
@property(nonatomic, readonly) NSString *label;

/** Whether the data channel can send messages in unreliable mode. */
@property(nonatomic, readonly) BOOL isReliable DEPRECATED_ATTRIBUTE;

/** Returns whether this data channel is ordered or not. */
@property(nonatomic, readonly) BOOL isOrdered;

/** Deprecated. Use maxPacketLifeTime. */
@property(nonatomic, readonly) NSUInteger maxRetransmitTime DEPRECATED_ATTRIBUTE;

/**
 * The length of the time window (in milliseconds) during which transmissions
 * and retransmissions may occur in unreliable mode.
 */
@property(nonatomic, readonly) uint16_t maxPacketLifeTime;

/**
 * The maximum number of retransmissions that are attempted in unreliable mode.
 */
@property(nonatomic, readonly) uint16_t maxRetransmits;

/**
 * The name of the sub-protocol used with this data channel, if any. Otherwise
 * this returns an empty string.
 */
@property(nonatomic, readonly) NSString *protocol;

/**
 * Returns whether this data channel was negotiated by the application or not.
 */
@property(nonatomic, readonly) BOOL isNegotiated;

/** Deprecated. Use channelId. */
@property(nonatomic, readonly) NSInteger streamId DEPRECATED_ATTRIBUTE;

/** The identifier for this data channel. */
@property(nonatomic, readonly) int channelId;

/** The state of the data channel. */
@property(nonatomic, readonly) RTCDataChannelState readyState;

/**
 * The number of bytes of application data that have been queued using
 * |sendData:| but that have not yet been transmitted to the network.
 */
@property(nonatomic, readonly) uint64_t bufferedAmount;

/** The delegate for this data channel. */
@property(nonatomic, weak) id<RTCDataChannelDelegate> delegate;

- (instancetype)init NS_UNAVAILABLE;

/** Closes the data channel. */
- (void)close;

/** Attempt to send |data| on this data channel's underlying data transport. */
- (BOOL)sendData:(RTCDataBuffer *)data;

@end

NS_ASSUME_NONNULL_END
