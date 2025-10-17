/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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

//===--- SILDefaultOverrideTable.cpp --------------------------------------===//
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
// This file defines the SILDefaultOverrideTable class, which is used to
// provide default override implementations of class routines which have been
// come to implement the same semantic class member that was previously
// implemented by a different routine.  As with SILDefaultWitnessTable, this
// type enables IRGen to generate metadata which in turn allows the runtime to
// instantiate vtables which contain these default overrides when they are
// needed: in the vtables of subclasses which were emitted prior to the
// replacement of the routine that implements the semantic member AND which
// already provided an override of the routine that previously implemented the
// semantic member.
//
//===----------------------------------------------------------------------===//

#include "language/SIL/SILDefaultOverrideTable.h"
#include "language/AST/ASTMangler.h"
#include "language/SIL/SILFunction.h"
#include "language/SIL/SILModule.h"

using namespace language;

SILDefaultOverrideTable::SILDefaultOverrideTable(SILModule &M,
                                                 SILLinkage Linkage,
                                                 const ClassDecl *decl)
    : module(M), linkage(Linkage), decl(decl), entries({}),
      state(State::Declared) {}

SILDefaultOverrideTable *
SILDefaultOverrideTable::declare(SILModule &module, SILLinkage linkage,
                                 const ClassDecl *decl) {
  auto *buffer = module.allocate<SILDefaultOverrideTable>(1);
  SILDefaultOverrideTable *table =
      ::new (buffer) SILDefaultOverrideTable(module, linkage, decl);

  table->registerWithModule();

  return table;
}

SILDefaultOverrideTable *SILDefaultOverrideTable::define(
    SILModule &module, SILLinkage linkage, const ClassDecl *decl,
    ArrayRef<SILDefaultOverrideTable::Entry> entries) {
  auto *buffer = module.allocate<SILDefaultOverrideTable>(1);
  SILDefaultOverrideTable *table =
      ::new (buffer) SILDefaultOverrideTable(module, linkage, decl);
  table->define(entries);

  table->registerWithModule();

  // Return the resulting default witness table.
  return table;
}

void SILDefaultOverrideTable::registerWithModule() {
  assert(module.DefaultOverrideTableMap.find(decl) ==
         module.DefaultOverrideTableMap.end());

  module.DefaultOverrideTableMap[decl] = this;
  module.defaultOverrideTables.push_back(this);
}

std::string SILDefaultOverrideTable::getUniqueName() const {
  Mangle::ASTMangler Mangler(decl->getASTContext());
  return Mangler.mangleTypeWithoutPrefix(
      decl->getDeclaredInterfaceType()->getCanonicalType());
}

void SILDefaultOverrideTable::define(ArrayRef<Entry> entries) {
  assert(state == State::Declared);
  state = State::Defined;

  this->entries = module.allocateCopy(entries);

  // Retain referenced functions.
  for (auto entry : getEntries()) {
    entry.impl->incrementRefCount();
  }
}

SILDefaultOverrideTable::~SILDefaultOverrideTable() {
  // Release referenced functions.
  for (auto entry : getEntries()) {
    entry.impl->decrementRefCount();
  }
}
