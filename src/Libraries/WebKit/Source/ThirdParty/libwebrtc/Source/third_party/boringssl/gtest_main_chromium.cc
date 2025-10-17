/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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

// Copyright 2017 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "base/functional/bind.h"
#include "base/test/launcher/unit_test_launcher.h"
#include "base/test/test_suite.h"
#include "testing/gtest/include/gtest/gtest.h"
#include "third_party/boringssl/src/crypto/test/gtest_main.h"

namespace {

class BoringSSLTestSuite : public base::TestSuite {
 public:
  BoringSSLTestSuite(int argc, char** argv) : TestSuite(argc, argv) {}

  void Initialize() override {
    TestSuite::Initialize();
    bssl::SetupGoogleTest();
  }
};

}  // namespace

int main(int argc, char** argv) {
  BoringSSLTestSuite test_suite(argc, argv);
  return base::LaunchUnitTests(
      argc, argv,
      base::BindOnce(&BoringSSLTestSuite::Run, base::Unretained(&test_suite)));
}
