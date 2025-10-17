/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
#ifndef RTC_BASE_ASYNC_DNS_RESOLVER_H_
#define RTC_BASE_ASYNC_DNS_RESOLVER_H_

#include <vector>

#include "api/async_dns_resolver.h"
#include "api/sequence_checker.h"
#include "api/task_queue/pending_task_safety_flag.h"
#include "rtc_base/ref_counted_object.h"
#include "rtc_base/system/rtc_export.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {
// This file contains a default implementation of
// webrtc::AsyncDnsResolverInterface, for use when there is no need for special
// treatment.

class AsyncDnsResolverResultImpl : public AsyncDnsResolverResult {
 public:
  bool GetResolvedAddress(int family, rtc::SocketAddress* addr) const override;
  // Returns error from resolver.
  int GetError() const override;

 private:
  friend class AsyncDnsResolver;
  RTC_NO_UNIQUE_ADDRESS webrtc::SequenceChecker sequence_checker_;
  rtc::SocketAddress addr_ RTC_GUARDED_BY(sequence_checker_);
  std::vector<rtc::IPAddress> addresses_ RTC_GUARDED_BY(sequence_checker_);
  int error_ RTC_GUARDED_BY(sequence_checker_);
};

class RTC_EXPORT AsyncDnsResolver : public AsyncDnsResolverInterface {
 public:
  AsyncDnsResolver();
  ~AsyncDnsResolver();
  // Start address resolution of the hostname in `addr`.
  void Start(const rtc::SocketAddress& addr,
             absl::AnyInvocable<void()> callback) override;
  // Start address resolution of the hostname in `addr` matching `family`.
  void Start(const rtc::SocketAddress& addr,
             int family,
             absl::AnyInvocable<void()> callback) override;
  const AsyncDnsResolverResult& result() const override;

 private:
  class State;
  ScopedTaskSafety safety_;          // To check for client going away
  rtc::scoped_refptr<State> state_;  // To check for "this" going away
  AsyncDnsResolverResultImpl result_;
  absl::AnyInvocable<void()> callback_;
};

}  // namespace webrtc
#endif  // RTC_BASE_ASYNC_DNS_RESOLVER_H_
