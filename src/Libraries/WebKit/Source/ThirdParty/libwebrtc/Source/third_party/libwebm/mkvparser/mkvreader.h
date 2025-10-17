/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 26, 2024.
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

// Copyright (c) 2010 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef MKVPARSER_MKVREADER_H_
#define MKVPARSER_MKVREADER_H_

#include <cstdio>

#include "mkvparser/mkvparser.h"

namespace mkvparser {

class MkvReader : public IMkvReader {
 public:
  MkvReader();
  explicit MkvReader(FILE* fp);
  virtual ~MkvReader();

  int Open(const char*);
  void Close();

  virtual int Read(long long position, long length, unsigned char* buffer);
  virtual int Length(long long* total, long long* available);

 private:
  MkvReader(const MkvReader&);
  MkvReader& operator=(const MkvReader&);

  // Determines the size of the file. This is called either by the constructor
  // or by the Open function depending on file ownership. Returns true on
  // success.
  bool GetFileSize();

  long long m_length;
  FILE* m_file;
  bool reader_owns_file_;
};

}  // namespace mkvparser

#endif  // MKVPARSER_MKVREADER_H_
