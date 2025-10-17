/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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

//===--- BlockList.h ---------------------------------------------*- C++ -*-===//
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
//
// This file defines some miscellaneous overloads of hash_value() and
// simple_display().
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_BASIC_BLOCKLIST_H
#define LANGUAGE_BASIC_BLOCKLIST_H

#include "language/Basic/Toolchain.h"
#include "toolchain/ADT/StringRef.h"

namespace language {

class SourceManager;

enum class BlockListAction: uint8_t {
  Undefined = 0,
#define BLOCKLIST_ACTION(NAME) NAME,
#include "BlockListAction.def"
};

enum class BlockListKeyKind: uint8_t {
  Undefined = 0,
  ModuleName,
  ProjectName
};

class BlockListStore {
public:
  struct Implementation;
  void addConfigureFilePath(StringRef path);
  bool hasBlockListAction(StringRef key, BlockListKeyKind keyKind,
                          BlockListAction action);
  BlockListStore(SourceManager &SM);
  ~BlockListStore();
private:
  Implementation &Impl;
};

} // namespace language

#endif // LANGUAGE_BASIC_BLOCKLIST_H
