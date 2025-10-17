/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 3, 2022.
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

//===---- llvm/Support/DataStream.h - Lazy bitcode streaming ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines DataStreamer, which fetches bytes of data from
// a stream source. It provides support for streaming (lazy reading) of
// data, e.g. bitcode
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_SUPPORT_DATASTREAM_H_
#define LLVM_SUPPORT_DATASTREAM_H_

#include <string>

namespace llvm {

class DataStreamer {
public:
  /// Fetch bytes [start-end) from the stream, and write them to the
  /// buffer pointed to by buf. Returns the number of bytes actually written.
  virtual size_t GetBytes(unsigned char *buf, size_t len) = 0;

  virtual ~DataStreamer();
};

DataStreamer *getDataFileStreamer(const std::string &Filename,
                                  std::string *Err);

}

#endif  // LLVM_SUPPORT_DATASTREAM_H_
