/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
#import "RTCLogging.h"

#include "rtc_base/logging.h"

rtc::LoggingSeverity RTCGetNativeLoggingSeverity(RTCLoggingSeverity severity) {
  switch (severity) {
    case RTCLoggingSeverityVerbose:
      return rtc::LS_VERBOSE;
    case RTCLoggingSeverityInfo:
      return rtc::LS_INFO;
    case RTCLoggingSeverityWarning:
      return rtc::LS_WARNING;
    case RTCLoggingSeverityError:
      return rtc::LS_ERROR;
    case RTCLoggingSeverityNone:
      return rtc::LS_NONE;
  }
}

void RTCLogEx(RTCLoggingSeverity severity, NSString* log_string) {
  if (log_string.length) {
    const char* utf8_string = log_string.UTF8String;
    RTC_LOG_V(RTCGetNativeLoggingSeverity(severity)) << utf8_string;
  }
}

void RTCSetMinDebugLogLevel(RTCLoggingSeverity severity) {
  rtc::LogMessage::LogToDebug(RTCGetNativeLoggingSeverity(severity));
}

NSString* RTCFileName(const char* file_path) {
  NSString* ns_file_path =
      [[NSString alloc] initWithBytesNoCopy:const_cast<char*>(file_path)
                                     length:strlen(file_path)
                                   encoding:NSUTF8StringEncoding
                               freeWhenDone:NO];
  return ns_file_path.lastPathComponent;
}
