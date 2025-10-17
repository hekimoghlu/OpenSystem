/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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

//===----- ParseableOutput.h - Helpers for parseable output ------- C++ -*-===//
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
///
/// \file
/// Helpers for emitting the compiler's parseable output.
///
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_PARSEABLEOUTPUT_H
#define LANGUAGE_PARSEABLEOUTPUT_H

#include "language/Basic/Toolchain.h"
#include "language/Basic/TaskQueue.h"
#include "language/Basic/FileTypes.h"

namespace language {

namespace parseable_output {

/// Quasi-PIDs are _negative_ PID-like unique keys used to
/// masquerade batch job constituents as (quasi)processes, when writing
/// parseable output to consumers that don't understand the idea of a batch
/// job. They are negative in order to avoid possibly colliding with real
/// PIDs (which are always positive). We start at -1000 here as a crude but
/// harmless hedge against colliding with an errno value that might slip
/// into the stream of real PIDs (say, due to a TaskQueue bug).
const int QUASI_PID_START = -1000;

struct CommandInput {
  std::string Path;
  CommandInput() {}
  CommandInput(StringRef Path) : Path(Path) {}
};

using OutputPair = std::pair<file_types::ID, std::string>;

/// A client-agnostic (e.g. either the compiler driver or `language-frontend`)
/// description of a task that is the subject of a parseable-output message.
struct DetailedTaskDescription {
  std::string Executable;
  SmallVector<std::string, 16> Arguments;
  std::string CommandLine;
  SmallVector<CommandInput, 4> Inputs;
  SmallVector<OutputPair, 8> Outputs;
};

/// Emits a "began" message to the given stream.
void emitBeganMessage(raw_ostream &os, StringRef Name,
                      DetailedTaskDescription TascDesc,
                      int64_t Pid, sys::TaskProcessInformation ProcInfo);

/// Emits a "finished" message to the given stream.
void emitFinishedMessage(raw_ostream &os, StringRef Name,
                         std::string Output, int ExitStatus,
                         int64_t Pid, sys::TaskProcessInformation ProcInfo);

/// Emits a "signalled" message to the given stream.
void emitSignalledMessage(raw_ostream &os, StringRef Name, StringRef ErrorMsg,
                          StringRef Output, std::optional<int> Signal,
                          int64_t Pid, sys::TaskProcessInformation ProcInfo);

/// Emits a "skipped" message to the given stream.
void emitSkippedMessage(raw_ostream &os, StringRef Name,
                        DetailedTaskDescription TascDesc);

} // end namespace parseable_output
} // end namespace language

#endif
