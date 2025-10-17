/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_PROTOBUF_UTILS_H_
#define MODULES_AUDIO_PROCESSING_TEST_PROTOBUF_UTILS_H_

#include <memory>
#include <sstream>  // no-presubmit-check TODO(webrtc:8982)

#include "rtc_base/protobuf_utils.h"

// Generated at build-time by the protobuf compiler.
#include "modules/audio_processing/debug.pb.h"

namespace webrtc {

// Allocates new memory in the unique_ptr to fit the raw message and returns the
// number of bytes read.
size_t ReadMessageBytesFromFile(FILE* file, std::unique_ptr<uint8_t[]>* bytes);

// Returns true on success, false on error or end-of-file.
bool ReadMessageFromFile(FILE* file, MessageLite* msg);

// Returns true on success, false on error or end of string stream.
bool ReadMessageFromString(
    std::stringstream* input,  // no-presubmit-check TODO(webrtc:8982)
    MessageLite* msg);

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_PROTOBUF_UTILS_H_
