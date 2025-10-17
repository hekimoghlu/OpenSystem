/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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

// Subset of rtc::LoggingSeverity.
typedef NS_ENUM(NSInteger, RTCLoggingSeverity) {
  RTCLoggingSeverityVerbose,
  RTCLoggingSeverityInfo,
  RTCLoggingSeverityWarning,
  RTCLoggingSeverityError,
  RTCLoggingSeverityNone,
};

// Wrapper for C++ RTC_LOG(sev) macros.
// Logs the log string to the webrtc logstream for the given severity.
RTC_EXTERN void RTCLogEx(RTCLoggingSeverity severity, NSString* log_string);

// Wrapper for rtc::LogMessage::LogToDebug.
// Sets the minimum severity to be logged to console.
RTC_EXTERN void RTCSetMinDebugLogLevel(RTCLoggingSeverity severity);

// Returns the filename with the path prefix removed.
RTC_EXTERN NSString* RTCFileName(const char* filePath);

// Some convenience macros.

#define RTCLogString(format, ...)                                           \
  [NSString stringWithFormat:@"(%@:%d %s): " format, RTCFileName(__FILE__), \
                             __LINE__, __FUNCTION__, ##__VA_ARGS__]

#define RTCLogFormat(severity, format, ...)                     \
  do {                                                          \
    NSString* log_string = RTCLogString(format, ##__VA_ARGS__); \
    RTCLogEx(severity, log_string);                             \
  } while (false)

#define RTCLogVerbose(format, ...) \
  RTCLogFormat(RTCLoggingSeverityVerbose, format, ##__VA_ARGS__)

#define RTCLogInfo(format, ...) \
  RTCLogFormat(RTCLoggingSeverityInfo, format, ##__VA_ARGS__)

#define RTCLogWarning(format, ...) \
  RTCLogFormat(RTCLoggingSeverityWarning, format, ##__VA_ARGS__)

#define RTCLogError(format, ...) \
  RTCLogFormat(RTCLoggingSeverityError, format, ##__VA_ARGS__)

#if !defined(NDEBUG)
#define RTCLogDebug(format, ...) RTCLogInfo(format, ##__VA_ARGS__)
#else
#define RTCLogDebug(format, ...) \
  do {                           \
  } while (false)
#endif

#define RTCLog(format, ...) RTCLogInfo(format, ##__VA_ARGS__)
