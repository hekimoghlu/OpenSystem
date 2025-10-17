/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 31, 2023.
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

// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef INCLUDE_WEBM_BUFFER_READER_H_
#define INCLUDE_WEBM_BUFFER_READER_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "./reader.h"
#include "./status.h"

/**
 \file
 A `Reader` implementation that reads from a `std::vector<std::uint8_t>`.
 */

namespace webm {

/**
 \addtogroup PUBLIC_API
 @{
 */

/**
 A simple reader that reads data from a buffer of bytes.
 */
class BufferReader : public Reader {
 public:
  /**
   Constructs a new, empty reader.
   */
  BufferReader() = default;

  /**
   Constructs a new reader by copying the provided reader into the new reader.

   \param other The source reader to copy.
   */
  BufferReader(const BufferReader& other) = default;

  /**
   Copies the provided reader into this reader.

   \param other The source reader to copy. May be equal to `*this`, in which
   case this is a no-op.
   \return `*this`.
   */
  BufferReader& operator=(const BufferReader& other) = default;

  /**
   Constructs a new reader by moving the provided reader into the new reader.

   \param other The source reader to move. After moving, it will be reset to an
   empty stream.
   */
  BufferReader(BufferReader&&);

  /**
   Moves the provided reader into this reader.

   \param other The source reader to move. After moving, it will be reset to an
   empty stream. May be equal to `*this`, in which case this is a no-op.
   \return `*this`.
   */
  BufferReader& operator=(BufferReader&&);

  /**
   Creates a new `BufferReader` populated with the provided bytes.

   \param bytes Bytes that are assigned to the internal buffer and used as the
   source which is read from.
   */
  BufferReader(std::initializer_list<std::uint8_t> bytes);

  /**
   Creates a new `BufferReader` populated with the provided data.

   \param vector A vector of bytes that is copied to the internal buffer and
   used as the source which is read from.
   */
  explicit BufferReader(const std::vector<std::uint8_t>& vector);

  /**
   Creates a new `BufferReader` populated with the provided data.

   \param vector A vector of bytes that is moved to the internal buffer and used
   as the source which is read from.
   */
  explicit BufferReader(std::vector<std::uint8_t>&& vector);

  /**
   Resets the reader to read from the given list of bytes, starting at the
   beginning.

   This makes `reader = {1, 2, 3};` effectively equivalent to `reader =
   BufferReader({1, 2, 3});`.

   \param bytes Bytes that are assigned to the internal buffer and used as the
   source which is read from.
   \return `*this`.
   */
  BufferReader& operator=(std::initializer_list<std::uint8_t> bytes);

  Status Read(std::size_t num_to_read, std::uint8_t* buffer,
              std::uint64_t* num_actually_read) override;

  Status Skip(std::uint64_t num_to_skip,
              std::uint64_t* num_actually_skipped) override;

  std::uint64_t Position() const override;

  /**
   Gets the total size of the buffer.
   */
  std::size_t size() const { return data_.size(); }

 private:
  // Stores the byte buffer from which data is read.
  std::vector<std::uint8_t> data_;

  // The position of the reader in the byte buffer.
  std::size_t pos_ = 0;
};

/**
 @}
 */

}  // namespace webm

#endif  // INCLUDE_WEBM_BUFFER_READER_H_
