/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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
#ifndef RTC_BASE_SYNCHRONIZATION_YIELD_POLICY_H_
#define RTC_BASE_SYNCHRONIZATION_YIELD_POLICY_H_

namespace rtc {
class YieldInterface {
 public:
  virtual ~YieldInterface() = default;
  virtual void YieldExecution() = 0;
};

// Sets the current thread-local yield policy while it's in scope and reverts
// to the previous policy when it leaves the scope.
class ScopedYieldPolicy final {
 public:
  explicit ScopedYieldPolicy(YieldInterface* policy);
  ScopedYieldPolicy(const ScopedYieldPolicy&) = delete;
  ScopedYieldPolicy& operator=(const ScopedYieldPolicy&) = delete;
  ~ScopedYieldPolicy();
  // Will yield as specified by the currently active thread-local yield policy
  // (which by default is a no-op).
  static void YieldExecution();

 private:
  YieldInterface* const previous_;
};

}  // namespace rtc

#endif  // RTC_BASE_SYNCHRONIZATION_YIELD_POLICY_H_
