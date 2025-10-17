/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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
#ifndef LIBWEBM_COMMON_FILE_UTIL_H_
#define LIBWEBM_COMMON_FILE_UTIL_H_

#include <stdint.h>

#include <string>

#include "mkvmuxer/mkvmuxertypes.h"  // LIBWEBM_DISALLOW_COPY_AND_ASSIGN()

namespace libwebm {

// Returns a temporary file name.
std::string GetTempFileName();

// Returns size of file specified by |file_name|, or 0 upon failure.
uint64_t GetFileSize(const std::string& file_name);

// Gets the contents file_name as a string. Returns false on error.
bool GetFileContents(const std::string& file_name, std::string* contents);

// Manages life of temporary file specified at time of construction. Deletes
// file upon destruction.
class TempFileDeleter {
 public:
  TempFileDeleter();
  explicit TempFileDeleter(std::string file_name) : file_name_(file_name) {}
  ~TempFileDeleter();
  const std::string& name() const { return file_name_; }

 private:
  std::string file_name_;
  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(TempFileDeleter);
};

}  // namespace libwebm

#endif  // LIBWEBM_COMMON_FILE_UTIL_H_
