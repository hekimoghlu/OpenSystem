/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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
#ifndef TEST_TESTSUPPORT_TEST_ARTIFACTS_H_
#define TEST_TESTSUPPORT_TEST_ARTIFACTS_H_

#include <stdint.h>
#include <stdlib.h>

#include <string>

namespace webrtc {
namespace test {

// If the test_artifacts_dir flag is set, returns true and copies the location
// of the dir to `out_dir`. Otherwise, return false.
bool GetTestArtifactsDir(std::string* out_dir);

// Writes a `length` bytes array `buffer` to `filename` in isolated output
// directory defined by swarming. If the file is existing, content will be
// appended. Otherwise a new file will be created. This function returns false
// if isolated output directory has not been defined, or `filename` indicates an
// invalid or non-writable file, or underlying file system errors.
bool WriteToTestArtifactsDir(const char* filename,
                             const uint8_t* buffer,
                             size_t length);

bool WriteToTestArtifactsDir(const char* filename, const std::string& content);

}  // namespace test
}  // namespace webrtc

#endif  // TEST_TESTSUPPORT_TEST_ARTIFACTS_H_
