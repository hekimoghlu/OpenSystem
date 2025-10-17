/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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

//===- raw_os_ostream.h - std::ostream adaptor for raw_ostream --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the raw_os_ostream class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RAW_OS_OSTREAM_H
#define LLVM_SUPPORT_RAW_OS_OSTREAM_H

#include "llvm/Support/raw_ostream.h"
#include <iosfwd>

namespace llvm {

/// raw_os_ostream - A raw_ostream that writes to an std::ostream.  This is a
/// simple adaptor class.  It does not check for output errors; clients should
/// use the underlying stream to detect errors.
class raw_os_ostream : public raw_ostream {
  std::ostream &OS;

  /// write_impl - See raw_ostream::write_impl.
  virtual void write_impl(const char *Ptr, size_t Size) LLVM_OVERRIDE;

  /// current_pos - Return the current position within the stream, not
  /// counting the bytes currently in the buffer.
  virtual uint64_t current_pos() const LLVM_OVERRIDE;

public:
  raw_os_ostream(std::ostream &O) : OS(O) {}
  ~raw_os_ostream();
};

} // end llvm namespace

#endif
