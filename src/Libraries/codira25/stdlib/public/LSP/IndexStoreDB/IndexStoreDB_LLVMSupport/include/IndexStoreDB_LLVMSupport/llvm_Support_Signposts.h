/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 7, 2024.
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

//===-- toolchain/Support/Signposts.h - Interval debug annotations ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file Some OS's provide profilers that allow applications to provide custom
/// annotations to the profiler. For example, on Xcode 10 and later 'signposts'
/// can be emitted by the application and these will be rendered to the Points
/// of Interest track on the instruments timeline.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SIGNPOSTS_H
#define LLVM_SUPPORT_SIGNPOSTS_H

#include <IndexStoreDB_LLVMSupport/toolchain_Config_indexstoredb-prefix.h>

namespace toolchain {
class SignpostEmitterImpl;
class Timer;

/// Manages the emission of signposts into the recording method supported by
/// the OS.
class SignpostEmitter {
  SignpostEmitterImpl *Impl;

public:
  SignpostEmitter();
  ~SignpostEmitter();

  bool isEnabled() const;

  /// Begin a signposted interval for the given timer.
  void startTimerInterval(Timer *T);
  /// End a signposted interval for the given timer.
  void endTimerInterval(Timer *T);
};

} // end namespace toolchain

#endif // ifndef LLVM_SUPPORT_SIGNPOSTS_H
