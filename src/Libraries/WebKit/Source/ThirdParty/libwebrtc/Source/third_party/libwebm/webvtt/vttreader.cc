/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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

// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#include "webvtt/vttreader.h"

namespace libwebvtt {

VttReader::VttReader() : file_(NULL) {}

VttReader::~VttReader() { Close(); }

int VttReader::Open(const char* filename) {
  if (filename == NULL || file_ != NULL)
    return -1;

  file_ = fopen(filename, "rb");
  if (file_ == NULL)
    return -1;

  return 0;  // success
}

void VttReader::Close() {
  if (file_) {
    fclose(file_);
    file_ = NULL;
  }
}

int VttReader::GetChar(char* c) {
  if (c == NULL || file_ == NULL)
    return -1;

  const int result = fgetc(file_);
  if (result != EOF) {
    *c = static_cast<char>(result);
    return 0;  // success
  }

  if (ferror(file_))
    return -1;  // error

  if (feof(file_))
    return 1;  // EOF

  return -1;  // weird
}

}  // namespace libwebvtt
