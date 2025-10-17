/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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
#ifndef NET_DCSCTP_TX_RETRANSMISSION_ERROR_COUNTER_H_
#define NET_DCSCTP_TX_RETRANSMISSION_ERROR_COUNTER_H_

#include <functional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "net/dcsctp/public/dcsctp_options.h"

namespace dcsctp {

// The RetransmissionErrorCounter is a simple counter with a limit, and when
// the limit is exceeded, the counter is exhausted and the connection will
// be closed. It's incremented on retransmission errors, such as the T3-RTX
// timer expiring, but also missing heartbeats and stream reset requests.
class RetransmissionErrorCounter {
 public:
  RetransmissionErrorCounter(absl::string_view log_prefix,
                             const DcSctpOptions& options)
      : log_prefix_(log_prefix), limit_(options.max_retransmissions) {}

  // Increments the retransmission timer. If the maximum error count has been
  // reached, `false` will be returned.
  bool Increment(absl::string_view reason);
  bool IsExhausted() const { return limit_.has_value() && counter_ > *limit_; }

  // Clears the retransmission errors.
  void Clear();

  // Returns its current value
  int value() const { return counter_; }

 private:
  const absl::string_view log_prefix_;
  const std::optional<int> limit_;
  int counter_ = 0;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_TX_RETRANSMISSION_ERROR_COUNTER_H_
