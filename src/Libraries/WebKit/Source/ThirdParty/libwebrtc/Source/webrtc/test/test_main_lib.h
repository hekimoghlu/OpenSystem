/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
#ifndef TEST_TEST_MAIN_LIB_H_
#define TEST_TEST_MAIN_LIB_H_

#include <memory>
#include <string>

namespace webrtc {

// Class to initialize test environment and run tests.
class TestMain {
 public:
  virtual ~TestMain() {}

  static std::unique_ptr<TestMain> Create();

  // Initializes test environment. Clients can add their own initialization
  // steps after call to this method and before running tests.
  // Returns 0 if initialization was successful and non 0 otherwise.
  virtual int Init() = 0;
  // Temporary for backward compatibility
  virtual int Init(int* argc, char* argv[]) = 0;

  // Runs test end return result error code. 0 - no errors.
  virtual int Run(int argc, char* argv[]) = 0;

 protected:
  TestMain() = default;

  std::string field_trials_;
};

}  // namespace webrtc

#endif  // TEST_TEST_MAIN_LIB_H_
