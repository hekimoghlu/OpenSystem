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

//===--- Util.h - Common Driver Utilities -----------------------*- C++ -*-===//
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

#ifndef LANGUAGE_DRIVER_UTIL_H
#define LANGUAGE_DRIVER_UTIL_H

#include "language/Basic/FileTypes.h"
#include "language/Basic/Toolchain.h"
#include "toolchain/ADT/SmallVector.h"

namespace toolchain {
namespace opt {
  class Arg;
} // end namespace opt
} // end namespace toolchain

namespace language {

namespace driver {
  /// An input argument from the command line and its inferred type.
  using InputPair = std::pair<file_types::ID, const toolchain::opt::Arg *>;
  /// Type used for a list of input arguments.
  using InputFileList = SmallVector<InputPair, 16>;

  enum class LinkKind {
    None,
    Executable,
    DynamicLibrary,
    StaticLibrary
  };

  /// Used by a Job to request a "filelist": a file containing a list of all
  /// input or output files of a certain type.
  ///
  /// The Compilation is responsible for generating this file before running
  /// the Job this info is attached to.
  struct FilelistInfo {
    enum class WhichFiles : unsigned {
      InputJobs,
      SourceInputActions,
      InputJobsAndSourceInputActions,
      Output,
      IndexUnitOutputPaths,
      /// Batch mode frontend invocations may have so many supplementary
      /// outputs that they don't comfortably fit as command-line arguments.
      /// In that case, add a FilelistInfo to record the path to the file.
      /// The type is ignored.
      SupplementaryOutput,
    };

    StringRef path;
    file_types::ID type;
    WhichFiles whichFiles;
  };

} // end namespace driver
} // end namespace language

#endif
