/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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

#import "RTCLogging.h"
#import "RTCMacros.h"

NS_ASSUME_NONNULL_BEGIN

typedef void (^RTCCallbackLoggerMessageHandler)(NSString *message);
typedef void (^RTCCallbackLoggerMessageAndSeverityHandler)(NSString *message,
                                                           RTCLoggingSeverity severity);

// This class intercepts WebRTC logs and forwards them to a registered block.
// This class is not threadsafe.
RTC_OBJC_EXPORT
@interface RTCCallbackLogger : NSObject

// The severity level to capture. The default is kRTCLoggingSeverityInfo.
@property(nonatomic, assign) RTCLoggingSeverity severity;

// The callback handler will be called on the same thread that does the
// logging, so if the logging callback can be slow it may be a good idea
// to implement dispatching to some other queue.
- (void)start:(nullable RTCCallbackLoggerMessageHandler)handler;
- (void)startWithMessageAndSeverityHandler:
        (nullable RTCCallbackLoggerMessageAndSeverityHandler)handler;

- (void)stop;

@end

NS_ASSUME_NONNULL_END
