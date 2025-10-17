/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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

//===--- BasicSourceInfo.cpp - Simple source information ------------------===//
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

#include "language/AST/ASTContext.h"
#include "language/AST/SourceFile.h"
#include "language/Basic/BasicSourceInfo.h"
#include "language/Basic/Defer.h"
#include "language/Basic/SourceManager.h"

using namespace language;

BasicSourceFileInfo::BasicSourceFileInfo(const SourceFile *SF)
    : SFAndIsFromSF(SF, true) {
  FilePath = SF->getFilename();
}

bool BasicSourceFileInfo::isFromSourceFile() const {
  return SFAndIsFromSF.getInt();
}

void BasicSourceFileInfo::populateWithSourceFileIfNeeded() {
  const auto *SF = SFAndIsFromSF.getPointer();
  if (!SF)
    return;
  LANGUAGE_DEFER {
    SFAndIsFromSF.setPointer(nullptr);
  };

  SourceManager &SM = SF->getASTContext().SourceMgr;

  if (FilePath.empty())
    return;
  auto stat = SM.getFileSystem()->status(FilePath);
  if (!stat)
    return;

  LastModified = stat->getLastModificationTime();
  FileSize = stat->getSize();

  if (SF->hasInterfaceHash()) {
    InterfaceHashIncludingTypeMembers = SF->getInterfaceHashIncludingTypeMembers();
    InterfaceHashExcludingTypeMembers = SF->getInterfaceHash();
  } else {
    // FIXME: Parse the file with EnableInterfaceHash option.
    InterfaceHashIncludingTypeMembers = Fingerprint::ZERO();
    InterfaceHashExcludingTypeMembers = Fingerprint::ZERO();
  }

  return;
}
