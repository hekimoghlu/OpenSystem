/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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
#include "rtc_base/byte_buffer.h"

#include <string.h>

namespace rtc {

ByteBufferWriter::ByteBufferWriter() : ByteBufferWriterT() {}

ByteBufferWriter::ByteBufferWriter(const uint8_t* bytes, size_t len)
    : ByteBufferWriterT(bytes, len) {}

ByteBufferReader::ByteBufferReader(rtc::ArrayView<const uint8_t> bytes) {
  Construct(bytes.data(), bytes.size());
}

ByteBufferReader::ByteBufferReader(const ByteBufferWriter& buf) {
  Construct(reinterpret_cast<const uint8_t*>(buf.Data()), buf.Length());
}

void ByteBufferReader::Construct(const uint8_t* bytes, size_t len) {
  bytes_ = bytes;
  size_ = len;
  start_ = 0;
  end_ = len;
}

bool ByteBufferReader::ReadUInt8(uint8_t* val) {
  if (!val)
    return false;

  return ReadBytes(val, 1);
}

bool ByteBufferReader::ReadUInt16(uint16_t* val) {
  if (!val)
    return false;

  uint16_t v;
  if (!ReadBytes(reinterpret_cast<uint8_t*>(&v), 2)) {
    return false;
  } else {
    *val = NetworkToHost16(v);
    return true;
  }
}

bool ByteBufferReader::ReadUInt24(uint32_t* val) {
  if (!val)
    return false;

  uint32_t v = 0;
  uint8_t* read_into = reinterpret_cast<uint8_t*>(&v);
  ++read_into;

  if (!ReadBytes(read_into, 3)) {
    return false;
  } else {
    *val = NetworkToHost32(v);
    return true;
  }
}

bool ByteBufferReader::ReadUInt32(uint32_t* val) {
  if (!val)
    return false;

  uint32_t v;
  if (!ReadBytes(reinterpret_cast<uint8_t*>(&v), 4)) {
    return false;
  } else {
    *val = NetworkToHost32(v);
    return true;
  }
}

bool ByteBufferReader::ReadUInt64(uint64_t* val) {
  if (!val)
    return false;

  uint64_t v;
  if (!ReadBytes(reinterpret_cast<uint8_t*>(&v), 8)) {
    return false;
  } else {
    *val = NetworkToHost64(v);
    return true;
  }
}

bool ByteBufferReader::ReadUVarint(uint64_t* val) {
  if (!val) {
    return false;
  }
  // Integers are deserialized 7 bits at a time, with each byte having a
  // continuation byte (msb=1) if there are more bytes to be read.
  uint64_t v = 0;
  for (int i = 0; i < 64; i += 7) {
    uint8_t byte;
    if (!ReadBytes(&byte, 1)) {
      return false;
    }
    // Read the first 7 bits of the byte, then offset by bits read so far.
    v |= (static_cast<uint64_t>(byte) & 0x7F) << i;
    // Return if the msb is not a continuation byte.
    if (byte < 0x80) {
      *val = v;
      return true;
    }
  }
  return false;
}

bool ByteBufferReader::ReadString(std::string* val, size_t len) {
  if (!val)
    return false;

  if (len > Length()) {
    return false;
  } else {
    val->append(reinterpret_cast<const char*>(bytes_ + start_), len);
    start_ += len;
    return true;
  }
}

bool ByteBufferReader::ReadStringView(absl::string_view* val, size_t len) {
  if (!val || len > Length())
    return false;
  *val = absl::string_view(reinterpret_cast<const char*>(bytes_ + start_), len);
  start_ += len;
  return true;
}

bool ByteBufferReader::ReadBytes(rtc::ArrayView<uint8_t> val) {
  if (val.size() == 0) {
    return true;
  }
  return ReadBytes(val.data(), val.size());
}

// Private function supporting the other Read* functions.
bool ByteBufferReader::ReadBytes(uint8_t* val, size_t len) {
  if (len > Length()) {
    return false;
  }
  if (len == 0) {
    return true;
  }
  memcpy(val, bytes_ + start_, len);
  start_ += len;
  return true;
}

bool ByteBufferReader::Consume(size_t size) {
  if (size > Length())
    return false;
  start_ += size;
  return true;
}

}  // namespace rtc
