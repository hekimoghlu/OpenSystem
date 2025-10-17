/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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

//===--- PrettyStackTrace.h - PrettyStackTrace for Transforms ---*- C++ -*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_SILOPTIMIZER_PASSMANAGER_PRETTYSTACKTRACE_H
#define LANGUAGE_SILOPTIMIZER_PASSMANAGER_PRETTYSTACKTRACE_H

#include "language/SIL/PrettyStackTrace.h"
#include "toolchain/Support/PrettyStackTrace.h"

namespace language {

class SILFunctionTransform;
class SILModuleTransform;

class PrettyStackTraceSILFunctionTransform
    : public PrettyStackTraceSILFunction {
  SILFunctionTransform *SFT;
  unsigned PassNumber;

public:
  PrettyStackTraceSILFunctionTransform(SILFunctionTransform *SFT,
                                       unsigned PassNumber);

  virtual void print(toolchain::raw_ostream &OS) const override;
};

class PrettyStackTraceSILModuleTransform : public toolchain::PrettyStackTraceEntry {
  SILModuleTransform *SMT;
  unsigned PassNumber;

public:
  PrettyStackTraceSILModuleTransform(SILModuleTransform *SMT,
                                     unsigned PassNumber)
      : SMT(SMT), PassNumber(PassNumber) {}
  virtual void print(toolchain::raw_ostream &OS) const override;
};

} // end namespace language

#endif
