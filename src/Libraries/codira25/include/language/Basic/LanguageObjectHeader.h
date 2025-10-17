/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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

//===--- LanguageObjectHeader.h - Defines LanguageObjectHeader ------------------===//
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
#ifndef LANGUAGE_BASIC_LANGUAGEOBJECTHEADER_H
#define LANGUAGE_BASIC_LANGUAGEOBJECTHEADER_H

#include "BridgedCodiraObject.h"

/// The C++ version of CodiraObject.
///
/// It is used for bridging the SIL core classes (e.g. SILFunction, SILNode,
/// etc.) with Codira.
/// For details see CodiraCompilerSources/README.md.
///
/// In C++ code, never use BridgedCodiraObject directly. LanguageObjectHeader has
/// the proper constructor, which avoids the header to be uninitialized.
struct LanguageObjectHeader : BridgedCodiraObject {
  LanguageObjectHeader(CodiraMetatype metatype) {
    this->metatype = metatype;
    this->refCounts = ~(uint64_t)0;
  }

  bool isBridged() const { return metatype != nullptr; }
};

#endif
