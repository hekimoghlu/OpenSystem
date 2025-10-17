/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 27, 2024.
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
#import "RTCCallbackLogger.h"

#include <memory>

#include "rtc_base/checks.h"
#include "rtc_base/log_sinks.h"
#include "rtc_base/logging.h"

class CallbackLogSink : public rtc::LogSink {
 public:
  CallbackLogSink(RTCCallbackLoggerMessageHandler callbackHandler)
      : callback_handler_(callbackHandler) {}

  void OnLogMessage(const std::string &message) override {
    if (callback_handler_) {
      callback_handler_([NSString stringWithUTF8String:message.c_str()]);
    }
  }

 private:
  RTCCallbackLoggerMessageHandler callback_handler_;
};

class CallbackWithSeverityLogSink : public rtc::LogSink {
 public:
  CallbackWithSeverityLogSink(RTCCallbackLoggerMessageAndSeverityHandler callbackHandler)
      : callback_handler_(callbackHandler) {}

  void OnLogMessage(const std::string& message) override { RTC_NOTREACHED(); }

  void OnLogMessage(const std::string& message, rtc::LoggingSeverity severity) override {
    if (callback_handler_) {
      RTCLoggingSeverity loggingSeverity = NativeSeverityToObjcSeverity(severity);
      callback_handler_([NSString stringWithUTF8String:message.c_str()], loggingSeverity);
    }
  }

 private:
  static RTCLoggingSeverity NativeSeverityToObjcSeverity(rtc::LoggingSeverity severity) {
    switch (severity) {
      case rtc::LS_VERBOSE:
        return RTCLoggingSeverityVerbose;
      case rtc::LS_INFO:
        return RTCLoggingSeverityInfo;
      case rtc::LS_WARNING:
        return RTCLoggingSeverityWarning;
      case rtc::LS_ERROR:
        return RTCLoggingSeverityError;
      case rtc::LS_NONE:
        return RTCLoggingSeverityNone;
    }
  }

  RTCCallbackLoggerMessageAndSeverityHandler callback_handler_;
};

@implementation RTCCallbackLogger {
  BOOL _hasStarted;
  std::unique_ptr<rtc::LogSink> _logSink;
}

@synthesize severity = _severity;

- (instancetype)init {
  self = [super init];
  if (self != nil) {
    _severity = RTCLoggingSeverityInfo;
  }
  return self;
}

- (void)dealloc {
  [self stop];
}

- (void)start:(nullable RTCCallbackLoggerMessageHandler)handler {
  if (_hasStarted) {
    return;
  }

  _logSink.reset(new CallbackLogSink(handler));

  rtc::LogMessage::AddLogToStream(_logSink.get(), [self rtcSeverity]);
  _hasStarted = YES;
}

- (void)startWithMessageAndSeverityHandler:
        (nullable RTCCallbackLoggerMessageAndSeverityHandler)handler {
  if (_hasStarted) {
    return;
  }

  _logSink.reset(new CallbackWithSeverityLogSink(handler));

  rtc::LogMessage::AddLogToStream(_logSink.get(), [self rtcSeverity]);
  _hasStarted = YES;
}

- (void)stop {
  if (!_hasStarted) {
    return;
  }
  RTC_DCHECK(_logSink);
  rtc::LogMessage::RemoveLogToStream(_logSink.get());
  _hasStarted = NO;
  _logSink.reset();
}

#pragma mark - Private

- (rtc::LoggingSeverity)rtcSeverity {
  switch (_severity) {
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

@end
