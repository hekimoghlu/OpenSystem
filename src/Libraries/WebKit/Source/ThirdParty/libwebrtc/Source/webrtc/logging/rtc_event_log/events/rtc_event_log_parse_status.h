/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_LOG_PARSE_STATUS_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_LOG_PARSE_STATUS_H_

#include <string>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"

#define RTC_PARSE_RETURN_ERROR(X)                                 \
  do {                                                            \
    return RtcEventLogParseStatus::Error(#X, __FILE__, __LINE__); \
  } while (0)

#define RTC_PARSE_CHECK_OR_RETURN(X)                                \
  do {                                                              \
    if (!(X))                                                       \
      return RtcEventLogParseStatus::Error(#X, __FILE__, __LINE__); \
  } while (0)

#define RTC_PARSE_CHECK_OR_RETURN_MESSAGE(X, M)                      \
  do {                                                               \
    if (!(X))                                                        \
      return RtcEventLogParseStatus::Error((M), __FILE__, __LINE__); \
  } while (0)

#define RTC_PARSE_CHECK_OR_RETURN_OP(OP, X, Y)                             \
  do {                                                                     \
    if (!((X)OP(Y)))                                                       \
      return RtcEventLogParseStatus::Error(#X #OP #Y, __FILE__, __LINE__); \
  } while (0)

#define RTC_PARSE_CHECK_OR_RETURN_EQ(X, Y) \
  RTC_PARSE_CHECK_OR_RETURN_OP(==, X, Y)

#define RTC_PARSE_CHECK_OR_RETURN_NE(X, Y) \
  RTC_PARSE_CHECK_OR_RETURN_OP(!=, X, Y)

#define RTC_PARSE_CHECK_OR_RETURN_LT(X, Y) RTC_PARSE_CHECK_OR_RETURN_OP(<, X, Y)

#define RTC_PARSE_CHECK_OR_RETURN_LE(X, Y) \
  RTC_PARSE_CHECK_OR_RETURN_OP(<=, X, Y)

#define RTC_PARSE_CHECK_OR_RETURN_GT(X, Y) RTC_PARSE_CHECK_OR_RETURN_OP(>, X, Y)

#define RTC_PARSE_CHECK_OR_RETURN_GE(X, Y) \
  RTC_PARSE_CHECK_OR_RETURN_OP(>=, X, Y)

#define RTC_PARSE_WARN_AND_RETURN_SUCCESS_IF(X, M) \
  do {                                             \
    if (X) {                                       \
      RTC_LOG(LS_WARNING) << (M);                  \
      return RtcEventLogParseStatus::Success();    \
    }                                              \
  } while (0)

#define RTC_RETURN_IF_ERROR(X)                         \
  do {                                                 \
    const RtcEventLogParseStatus _rtc_parse_status(X); \
    if (!_rtc_parse_status.ok()) {                     \
      return _rtc_parse_status;                        \
    }                                                  \
  } while (0)

// TODO(terelius): Compared to a generic 'Status' class, this
// class allows us additional information about the context
// in which the error occurred. This is currently limited to
// the source location (file and line), but we plan on adding
// information about the event and field name being parsed.
// If/when we start using absl::Status in WebRTC, consider
// whether payloads would be an appropriate alternative.
class RtcEventLogParseStatus {
  template <typename T>
  friend class RtcEventLogParseStatusOr;

 public:
  static RtcEventLogParseStatus Success() { return RtcEventLogParseStatus(); }
  static RtcEventLogParseStatus Error(absl::string_view error,
                                      absl::string_view file,
                                      int line) {
    return RtcEventLogParseStatus(error, file, line);
  }

  bool ok() const { return error_.empty(); }
  ABSL_DEPRECATED("Use ok() instead") explicit operator bool() const {
    return ok();
  }

  std::string message() const { return error_; }

 private:
  RtcEventLogParseStatus() : error_() {}
  RtcEventLogParseStatus(absl::string_view error,
                         absl::string_view file,
                         int line)
      : error_(std::string(error) + " (" + std::string(file) + ": " +
               std::to_string(line) + ")") {}

  std::string error_;
};

template <typename T>
class RtcEventLogParseStatusOr {
 public:
  RtcEventLogParseStatusOr(RtcEventLogParseStatus status)  // NOLINT
      : status_(status), value_() {}
  RtcEventLogParseStatusOr(const T& value)  // NOLINT
      : status_(), value_(value) {}

  bool ok() const { return status_.ok(); }

  std::string message() const { return status_.message(); }

  RtcEventLogParseStatus status() const { return status_; }

  const T& value() const {
    RTC_DCHECK(ok());
    return value_;
  }

  T& value() {
    RTC_DCHECK(ok());
    return value_;
  }

  static RtcEventLogParseStatusOr Error(absl::string_view error,
                                        absl::string_view file,
                                        int line) {
    return RtcEventLogParseStatusOr(error, file, line);
  }

 private:
  RtcEventLogParseStatusOr() : status_() {}
  RtcEventLogParseStatusOr(absl::string_view error,
                           absl::string_view file,
                           int line)
      : status_(error, file, line), value_() {}

  RtcEventLogParseStatus status_;
  T value_;
};

#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_LOG_PARSE_STATUS_H_
