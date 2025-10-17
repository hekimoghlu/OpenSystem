/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 12, 2022.
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
#include "modules/audio_processing/test/protobuf_utils.h"

#include <memory>

#include "rtc_base/system/arch.h"

namespace {
// Allocates new memory in the memory owned by the unique_ptr to fit the raw
// message and returns the number of bytes read when having a string stream as
// input.
size_t ReadMessageBytesFromString(std::stringstream* input,
                                  std::unique_ptr<uint8_t[]>* bytes) {
  int32_t size = 0;
  input->read(reinterpret_cast<char*>(&size), sizeof(int32_t));
  int32_t size_read = input->gcount();
  if (size_read != sizeof(int32_t))
    return 0;
  if (size <= 0)
    return 0;

  *bytes = std::make_unique<uint8_t[]>(size);
  input->read(reinterpret_cast<char*>(bytes->get()),
              size * sizeof((*bytes)[0]));
  size_read = input->gcount();
  return size_read == size ? size : 0;
}
}  // namespace

namespace webrtc {

size_t ReadMessageBytesFromFile(FILE* file, std::unique_ptr<uint8_t[]>* bytes) {
// The "wire format" for the size is little-endian. Assume we're running on
// a little-endian machine.
#ifndef WEBRTC_ARCH_LITTLE_ENDIAN
#error "Need to convert messsage from little-endian."
#endif
  int32_t size = 0;
  if (fread(&size, sizeof(size), 1, file) != 1)
    return 0;
  if (size <= 0)
    return 0;

  *bytes = std::make_unique<uint8_t[]>(size);
  return fread(bytes->get(), sizeof((*bytes)[0]), size, file);
}

// Returns true on success, false on error or end-of-file.
bool ReadMessageFromFile(FILE* file, MessageLite* msg) {
  std::unique_ptr<uint8_t[]> bytes;
  size_t size = ReadMessageBytesFromFile(file, &bytes);
  if (!size)
    return false;

  msg->Clear();
  return msg->ParseFromArray(bytes.get(), size);
}

// Returns true on success, false on error or end of string stream.
bool ReadMessageFromString(std::stringstream* input, MessageLite* msg) {
  std::unique_ptr<uint8_t[]> bytes;
  size_t size = ReadMessageBytesFromString(input, &bytes);
  if (!size)
    return false;

  msg->Clear();
  return msg->ParseFromArray(bytes.get(), size);
}

}  // namespace webrtc
