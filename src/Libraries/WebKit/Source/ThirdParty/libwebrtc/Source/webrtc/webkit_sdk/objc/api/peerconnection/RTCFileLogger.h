/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 9, 2025.
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

typedef NS_ENUM(NSUInteger, RTCFileLoggerSeverity) {
  RTCFileLoggerSeverityVerbose,
  RTCFileLoggerSeverityInfo,
  RTCFileLoggerSeverityWarning,
  RTCFileLoggerSeverityError
};

typedef NS_ENUM(NSUInteger, RTCFileLoggerRotationType) {
  RTCFileLoggerTypeCall,
  RTCFileLoggerTypeApp,
};

NS_ASSUME_NONNULL_BEGIN

// This class intercepts WebRTC logs and saves them to a file. The file size
// will not exceed the given maximum bytesize. When the maximum bytesize is
// reached, logs are rotated according to the rotationType specified.
// For kRTCFileLoggerTypeCall, logs from the beginning and the end
// are preserved while the middle section is overwritten instead.
// For kRTCFileLoggerTypeApp, the oldest log is overwritten.
// This class is not threadsafe.
RTC_OBJC_EXPORT
@interface RTCFileLogger : NSObject

// The severity level to capture. The default is kRTCFileLoggerSeverityInfo.
@property(nonatomic, assign) RTCFileLoggerSeverity severity;

// The rotation type for this file logger. The default is
// kRTCFileLoggerTypeCall.
@property(nonatomic, readonly) RTCFileLoggerRotationType rotationType;

// Disables buffering disk writes. Should be set before |start|. Buffering
// is enabled by default for performance.
@property(nonatomic, assign) BOOL shouldDisableBuffering;

// Default constructor provides default settings for dir path, file size and
// rotation type.
- (instancetype)init;

// Create file logger with default rotation type.
- (instancetype)initWithDirPath:(NSString *)dirPath maxFileSize:(NSUInteger)maxFileSize;

- (instancetype)initWithDirPath:(NSString *)dirPath
                    maxFileSize:(NSUInteger)maxFileSize
                   rotationType:(RTCFileLoggerRotationType)rotationType NS_DESIGNATED_INITIALIZER;

// Starts writing WebRTC logs to disk if not already started. Overwrites any
// existing file(s).
- (void)start;

// Stops writing WebRTC logs to disk. This method is also called on dealloc.
- (void)stop;

// Returns the current contents of the logs, or nil if start has been called
// without a stop.
- (nullable NSData *)logData;

@end

NS_ASSUME_NONNULL_END
