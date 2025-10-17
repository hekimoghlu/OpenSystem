/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 20, 2023.
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
#ifndef TEST_TESTSUPPORT_FILE_UTILS_OVERRIDE_H_
#define TEST_TESTSUPPORT_FILE_UTILS_OVERRIDE_H_

#include <string>

#include "absl/strings/string_view.h"

namespace webrtc {
namespace test {
namespace internal {

// Returns the absolute path to the output directory where log files and other
// test artifacts should be put. The output directory is generally a directory
// named "out" at the project root. This root is assumed to be two levels above
// where the test binary is located; this is because tests execute in a dir
// out/Whatever relative to the project root. This convention is also followed
// in Chromium.
//
// The exception is Android where we use /sdcard/ instead.
//
// If symbolic links occur in the path they will be resolved and the actual
// directory will be returned.
//
// Returns the path WITH a trailing path delimiter. If the project root is not
// found, the current working directory ("./") is returned as a fallback.
std::string OutputPath();

// Gets the current working directory for the executing program.
// Returns "./" if for some reason it is not possible to find the working
// directory.
std::string WorkingDir();

// Returns a full path to a resource file in the resources_dir dir.
//
// Arguments:
//    name - Name of the resource file. If a plain filename (no directory path)
//           is supplied, the file is assumed to be located in resources/
//           If a directory path is prepended to the filename, a subdirectory
//           hierarchy reflecting that path is assumed to be present.
//    extension - File extension, without the dot, i.e. "bmp" or "yuv".
std::string ResourcePath(absl::string_view name, absl::string_view extension);

}  // namespace internal
}  // namespace test
}  // namespace webrtc

#endif  // TEST_TESTSUPPORT_FILE_UTILS_OVERRIDE_H_
