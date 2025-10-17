/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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

//===--- PrettyStackTrace.h - PrettyStackTrace for the Driver ---*- C++ -*-===//
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

#ifndef LANGUAGE_DRIVER_PRETTYSTACKTRACE_H
#define LANGUAGE_DRIVER_PRETTYSTACKTRACE_H

#include "language/Basic/FileTypes.h"
#include "toolchain/Support/PrettyStackTrace.h"

namespace language {
namespace driver {

class Action;
class Job;
class CommandOutput;

class PrettyStackTraceDriverAction : public toolchain::PrettyStackTraceEntry {
  const Action *TheAction;
  const char *Description;

public:
  PrettyStackTraceDriverAction(const char *desc, const Action *A)
      : TheAction(A), Description(desc) {}
  void print(toolchain::raw_ostream &OS) const override;
};

class PrettyStackTraceDriverJob : public toolchain::PrettyStackTraceEntry {
  const Job *TheJob;
  const char *Description;

public:
  PrettyStackTraceDriverJob(const char *desc, const Job *A)
      : TheJob(A), Description(desc) {}
  void print(toolchain::raw_ostream &OS) const override;
};

class PrettyStackTraceDriverCommandOutput : public toolchain::PrettyStackTraceEntry {
  const CommandOutput *TheCommandOutput;
  const char *Description;

public:
  PrettyStackTraceDriverCommandOutput(const char *desc, const CommandOutput *A)
      : TheCommandOutput(A), Description(desc) {}
  void print(toolchain::raw_ostream &OS) const override;
};

class PrettyStackTraceDriverCommandOutputAddition
    : public toolchain::PrettyStackTraceEntry {
  const CommandOutput *TheCommandOutput;
  StringRef PrimaryInput;
  file_types::ID NewOutputType;
  StringRef NewOutputName;
  const char *Description;

public:
  PrettyStackTraceDriverCommandOutputAddition(const char *desc,
                                              const CommandOutput *A,
                                              StringRef Primary,
                                              file_types::ID type,
                                              StringRef New)
      : TheCommandOutput(A), PrimaryInput(Primary), NewOutputType(type),
        NewOutputName(New), Description(desc) {}
  void print(toolchain::raw_ostream &OS) const override;
};
}
}

#endif
