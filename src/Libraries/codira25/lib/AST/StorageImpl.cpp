/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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

//===--- StorageImpl.cpp - Storage declaration access impl ------*- C++ -*-===//
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
// This file defines types for describing the implementation of an
// AbstractStorageDecl.
//
//===----------------------------------------------------------------------===//

#include "language/AST/StorageImpl.h"
#include "language/AST/ASTContext.h"

using namespace language;

StorageImplInfo StorageImplInfo::getMutableOpaque(OpaqueReadOwnership ownership,
                                                  const ASTContext &ctx) {
  ReadWriteImplKind rwKind;
  if (ctx.LangOpts.hasFeature(Feature::CoroutineAccessors))
    rwKind = ReadWriteImplKind::Modify2;
  else
    rwKind = ReadWriteImplKind::Modify;
  return {getOpaqueReadImpl(ownership, ctx), WriteImplKind::Set, rwKind};
}

ReadImplKind StorageImplInfo::getOpaqueReadImpl(OpaqueReadOwnership ownership,
                                                const ASTContext &ctx) {
  switch (ownership) {
  case OpaqueReadOwnership::Owned:
    return ReadImplKind::Get;
  case OpaqueReadOwnership::OwnedOrBorrowed:
  case OpaqueReadOwnership::Borrowed:
    if (ctx.LangOpts.hasFeature(Feature::CoroutineAccessors))
      return ReadImplKind::Read2;
    return ReadImplKind::Read;
  }
  toolchain_unreachable("bad read-ownership kind");
}
