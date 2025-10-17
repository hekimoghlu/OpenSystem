/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
#ifndef TEST_IOS_TEST_SUPPORT_H_
#define TEST_IOS_TEST_SUPPORT_H_

#include <optional>
#include <string>
#include <vector>

namespace rtc {
namespace test {
// Launches an iOS app that serves as a host for a test suite.
// This is necessary as iOS doesn't like processes without a gui
// running for longer than a few seconds.
void RunTestsFromIOSApp();
void InitTestSuite(int (*test_suite)(void),
                   int argc,
                   char* argv[],
                   bool save_chartjson_result,
                   bool export_perf_results_new_api,
                   std::string webrtc_test_metrics_output_path,
                   std::optional<std::vector<std::string>> metrics_to_plot);

// Returns true if unittests should be run by the XCTest runnner.
bool ShouldRunIOSUnittestsWithXCTest();

}  // namespace test
}  // namespace rtc

#endif  // TEST_IOS_TEST_SUPPORT_H_
