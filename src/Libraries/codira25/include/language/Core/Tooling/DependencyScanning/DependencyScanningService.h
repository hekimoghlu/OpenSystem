/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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

//===- DependencyScanningService.h - clang-scan-deps service ===-*- C++ -*-===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_CORE_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H
#define LANGUAGE_CORE_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H

#include "language/Core/Tooling/DependencyScanning/DependencyScanningFilesystem.h"
#include "language/Core/Tooling/DependencyScanning/InProcessModuleCache.h"
#include "toolchain/ADT/BitmaskEnum.h"
#include "toolchain/Support/Chrono.h"

namespace language::Core {
namespace tooling {
namespace dependencies {

/// The mode in which the dependency scanner will operate to find the
/// dependencies.
enum class ScanningMode {
  /// This mode is used to compute the dependencies by running the preprocessor
  /// over the source files.
  CanonicalPreprocessing,

  /// This mode is used to compute the dependencies by running the preprocessor
  /// with special kind of lexing after scanning header and source files to get
  /// the minimum necessary preprocessor directives for evaluating includes.
  DependencyDirectivesScan,
};

/// The format that is output by the dependency scanner.
enum class ScanningOutputFormat {
  /// This is the Makefile compatible dep format. This will include all of the
  /// deps necessary for an implicit modules build, but won't include any
  /// intermodule dependency information.
  Make,

  /// This outputs the full clang module dependency graph suitable for use for
  /// explicitly building modules.
  Full,

  /// This outputs the dependency graph for standard c++ modules in P1689R5
  /// format.
  P1689,
};

#define DSS_LAST_BITMASK_ENUM(Id)                                              \
  LLVM_MARK_AS_BITMASK_ENUM(Id), All = toolchain::NextPowerOf2(Id) - 1

enum class ScanningOptimizations {
  None = 0,

  /// Remove unused header search paths including header maps.
  HeaderSearch = 1,

  /// Remove warnings from system modules.
  SystemWarnings = (1 << 1),

  /// Remove unused -ivfsoverlay arguments.
  VFS = (1 << 2),

  /// Canonicalize -D and -U options.
  Macros = (1 << 3),

  /// Ignore the compiler's working directory if it is safe.
  IgnoreCWD = (1 << 4),

  DSS_LAST_BITMASK_ENUM(IgnoreCWD),

  // The build system needs to be aware that the current working
  // directory is ignored. Without a good way of notifying the build
  // system, it is less risky to default to off.
  Default = All & (~IgnoreCWD)
};

#undef DSS_LAST_BITMASK_ENUM

/// The dependency scanning service contains shared configuration and state that
/// is used by the individual dependency scanning workers.
class DependencyScanningService {
public:
  DependencyScanningService(
      ScanningMode Mode, ScanningOutputFormat Format,
      ScanningOptimizations OptimizeArgs = ScanningOptimizations::Default,
      bool EagerLoadModules = false, bool TraceVFS = false,
      std::time_t BuildSessionTimestamp =
          toolchain::sys::toTimeT(std::chrono::system_clock::now()));

  ScanningMode getMode() const { return Mode; }

  ScanningOutputFormat getFormat() const { return Format; }

  ScanningOptimizations getOptimizeArgs() const { return OptimizeArgs; }

  bool shouldEagerLoadModules() const { return EagerLoadModules; }

  bool shouldTraceVFS() const { return TraceVFS; }

  DependencyScanningFilesystemSharedCache &getSharedCache() {
    return SharedCache;
  }

  ModuleCacheEntries &getModuleCacheEntries() { return ModCacheEntries; }

  std::time_t getBuildSessionTimestamp() const { return BuildSessionTimestamp; }

private:
  const ScanningMode Mode;
  const ScanningOutputFormat Format;
  /// Whether to optimize the modules' command-line arguments.
  const ScanningOptimizations OptimizeArgs;
  /// Whether to set up command-lines to load PCM files eagerly.
  const bool EagerLoadModules;
  /// Whether to trace VFS accesses.
  const bool TraceVFS;
  /// The global file system cache.
  DependencyScanningFilesystemSharedCache SharedCache;
  /// The global module cache entries.
  ModuleCacheEntries ModCacheEntries;
  /// The build session timestamp.
  std::time_t BuildSessionTimestamp;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace language::Core

#endif // LANGUAGE_CORE_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGSERVICE_H
