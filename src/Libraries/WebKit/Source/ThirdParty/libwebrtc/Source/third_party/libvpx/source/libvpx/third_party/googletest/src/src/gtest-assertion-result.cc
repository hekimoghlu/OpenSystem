/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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

// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// The Google C++ Testing and Mocking Framework (Google Test)
//
// This file defines the AssertionResult type.

#include "gtest/gtest-assertion-result.h"

#include <string>
#include <utility>

#include "gtest/gtest-message.h"

namespace testing {

// AssertionResult constructors.
// Used in EXPECT_TRUE/FALSE(assertion_result).
AssertionResult::AssertionResult(const AssertionResult& other)
    : success_(other.success_),
      message_(other.message_.get() != nullptr
                   ? new ::std::string(*other.message_)
                   : static_cast< ::std::string*>(nullptr)) {}

// Swaps two AssertionResults.
void AssertionResult::swap(AssertionResult& other) {
  using std::swap;
  swap(success_, other.success_);
  swap(message_, other.message_);
}

// Returns the assertion's negation. Used with EXPECT/ASSERT_FALSE.
AssertionResult AssertionResult::operator!() const {
  AssertionResult negation(!success_);
  if (message_.get() != nullptr) negation << *message_;
  return negation;
}

// Makes a successful assertion result.
AssertionResult AssertionSuccess() { return AssertionResult(true); }

// Makes a failed assertion result.
AssertionResult AssertionFailure() { return AssertionResult(false); }

// Makes a failed assertion result with the given failure message.
// Deprecated; use AssertionFailure() << message.
AssertionResult AssertionFailure(const Message& message) {
  return AssertionFailure() << message;
}

}  // namespace testing
