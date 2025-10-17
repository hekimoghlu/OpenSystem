/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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
#ifndef TEST_TEST_FLAGS_H_
#define TEST_TEST_FLAGS_H_

#include <string>
#include <vector>

#include "absl/flags/declare.h"

ABSL_DECLARE_FLAG(std::string, force_fieldtrials);
ABSL_DECLARE_FLAG(std::vector<std::string>, plot);
ABSL_DECLARE_FLAG(std::string, isolated_script_test_perf_output);
ABSL_DECLARE_FLAG(std::string, webrtc_test_metrics_output_path);
ABSL_DECLARE_FLAG(bool, export_perf_results_new_api);
ABSL_DECLARE_FLAG(bool, webrtc_quick_perf_test);

#endif  // TEST_TEST_FLAGS_H_
