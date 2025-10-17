/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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

//===- llvm/Support/Host.h - Host machine characteristics --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Methods for querying the nature of the host machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_HOST_H
#define LLVM_SYSTEM_HOST_H

#include "llvm/ADT/StringMap.h"
#include <string>

namespace llvm {
namespace sys {

  inline bool isLittleEndianHost() {
    union {
      int i;
      char c;
    };
    i = 1;
    return c;
  }

  inline bool isBigEndianHost() {
    return !isLittleEndianHost();
  }

  /// getDefaultTargetTriple() - Return the default target triple the compiler
  /// has been configured to produce code for.
  ///
  /// The target triple is a string in the format of:
  ///   CPU_TYPE-VENDOR-OPERATING_SYSTEM
  /// or
  ///   CPU_TYPE-VENDOR-KERNEL-OPERATING_SYSTEM
  std::string getDefaultTargetTriple();

  /// getHostCPUName - Get the LLVM name for the host CPU. The particular format
  /// of the name is target dependent, and suitable for passing as -mcpu to the
  /// target which matches the host.
  ///
  /// \return - The host CPU name, or empty if the CPU could not be determined.
  std::string getHostCPUName();

  /// getHostCPUFeatures - Get the LLVM names for the host CPU features.
  /// The particular format of the names are target dependent, and suitable for
  /// passing as -mattr to the target which matches the host.
  ///
  /// \param Features - A string mapping feature names to either
  /// true (if enabled) or false (if disabled). This routine makes no guarantees
  /// about exactly which features may appear in this map, except that they are
  /// all valid LLVM feature names.
  ///
  /// \return - True on success.
  bool getHostCPUFeatures(StringMap<bool> &Features);
}
}

#endif
