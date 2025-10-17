/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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

//===--- IndexRecord.h - Entry point for recording index data ---*- C++ -*-===//
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

#ifndef LANGUAGE_INDEX_INDEXRECORD_H
#define LANGUAGE_INDEX_INDEXRECORD_H

#include "language/Basic/Toolchain.h"
#include "language/Basic/PathRemapper.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/StringRef.h"

namespace language {
class DependencyTracker;
class ModuleDecl;
class SourceFile;

namespace index {

/// Index the given source file and store the results to \p indexStorePath.
///
/// \param primarySourceFile The source file to index.
///
/// \param indexUnitToken A unique identifier for this translation unit in the
/// form of a file path.
///
/// \param indexStorePath The location to write the indexing data to.
///
/// \param indexClangModules If true, emit index data for imported clang modules
/// (pcms).
///
/// \param indexSystemModules If true, emit index data for imported serialized
/// language system modules.
///
/// \param skipStdlib If indexing system modules, don't index the standard
/// library.
///
/// \param includeLocals If true, emit index data for local definitions and
/// references.
///
/// \param isDebugCompilation true for non-optimized compiler invocation.
///
/// \param targetTriple The target for this compilation.
///
/// \param dependencyTracker The set of dependencies seen while building.
///
/// \param pathRemapper Remapper to use for paths in index data.
bool indexAndRecord(SourceFile *primarySourceFile, StringRef indexUnitToken,
                    StringRef indexStorePath, bool indexClangModules,
                    bool indexSystemModules, bool skipStdlib,
                    bool includeLocals, bool isDebugCompilation,
                    bool isExplicitModuleBuild, StringRef targetTriple,
                    const DependencyTracker &dependencyTracker,
                    const PathRemapper &pathRemapper);

/// Index the given module and store the results to \p indexStorePath.
///
/// \param module The module to index.
///
/// \param indexUnitTokens A list of unique identifiers for the index units to
/// be written. This may either be one unit per source file of \p module, or it
/// may be a single unit, in which case all the index information will be
/// combined into a single unit.
///
/// \param moduleUnitToken A unique identifier for this module unit in the form
/// of a file path. Only used if \p indexUnitTokens are specified for each
/// source file, otherwise the single \p indexUnitTokens value is used instead.
///
/// \param indexStorePath The location to write the indexing data to.
///
/// \param indexClangModules If true, emit index data for imported clang modules
/// (pcms).
///
/// \param indexSystemModules If true, emit index data for imported serialized
/// language system modules.
///
/// \param skipStdlib If indexing system modules, don't index the standard
/// library.
///
/// \param includeLocals If true, emit index data for local definitions and
/// references.
///
/// \param isDebugCompilation true for non-optimized compiler invocation.
///
/// \param targetTriple The target for this compilation.
///
/// \param dependencyTracker The set of dependencies seen while building.
///
/// \param pathRemapper Remapper to use for paths in index data.
bool indexAndRecord(ModuleDecl *module, ArrayRef<std::string> indexUnitTokens,
                    StringRef moduleUnitToken, StringRef indexStorePath,
                    bool indexClangModules, bool indexSystemModules,
                    bool skipStdlib, bool includeLocals,
                    bool isDebugCompilation, bool isExplicitModuleBuild,
                    StringRef targetTriple,
                    const DependencyTracker &dependencyTracker,
                    const PathRemapper &pathRemapper);
// FIXME: indexUnitTokens could be StringRef, but that creates an impedance
// mismatch in the caller.

} // end namespace index
} // end namespace language

#endif // LANGUAGE_INDEX_INDEXRECORD_H
