/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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

//===--- Symbol.cpp -------------------------------------------------------===//
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

#include <IndexStoreDB_Core/Symbol.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_STLExtras.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_StringSwitch.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_ErrorHandling.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_raw_ostream.h>

using namespace IndexStoreDB;

bool SymbolInfo::isCallable() const {
  switch (Kind) {
  case SymbolKind::Function:
  case SymbolKind::InstanceMethod:
  case SymbolKind::ClassMethod:
  case SymbolKind::StaticMethod:
  case SymbolKind::Constructor:
  case SymbolKind::Destructor:
  case SymbolKind::ConversionFunction:
    return true;
  default:
    return false;
  }
}

bool SymbolInfo::preferDeclarationAsCanonical() const {
  if (Lang == SymbolLanguage::ObjC) {
    return Kind == SymbolKind::Class ||
      Kind == SymbolKind::Extension ||
      Kind == SymbolKind::InstanceProperty ||
      Kind == SymbolKind::ClassProperty;
  }
  return false;
}

bool SymbolInfo::includeInGlobalNameSearch() const {
  // Codira extensions don't have their own name, exclude them from global name search.
  // You can always lookup the class name and then find the class symbol extensions.
  if (Kind == SymbolKind::Extension && Lang == SymbolLanguage::Codira)
    return false;
  return true;
}

const char *IndexStoreDB::getSymbolKindString(SymbolKind kind) {
  switch (kind) {
  case SymbolKind::Unknown: return "unknown";
  case SymbolKind::Module: return "module";
  case SymbolKind::Macro: return "macro";
  case SymbolKind::Enum: return "enum";
  case SymbolKind::Struct: return "struct";
  case SymbolKind::Class: return "class";
  case SymbolKind::Protocol: return "protocol";
  case SymbolKind::Extension: return "extension";
  case SymbolKind::Union: return "union";
  case SymbolKind::TypeAlias: return "typealias";
  case SymbolKind::Function: return "function";
  case SymbolKind::Variable: return "variable";
  case SymbolKind::Field: return "field";
  case SymbolKind::Parameter: return "parameter";
  case SymbolKind::EnumConstant: return "enumerator";
  case SymbolKind::InstanceMethod: return "instance-method";
  case SymbolKind::ClassMethod: return "class-method";
  case SymbolKind::StaticMethod: return "static-method";
  case SymbolKind::InstanceProperty: return "instance-property";
  case SymbolKind::ClassProperty: return "class-property";
  case SymbolKind::StaticProperty: return "static-property";
  case SymbolKind::Constructor: return "constructor";
  case SymbolKind::Destructor: return "destructor";
  case SymbolKind::ConversionFunction: return "conversion-fn";
  case SymbolKind::Namespace: return "namespace";
  case SymbolKind::NamespaceAlias: return "namespace-alias";
  case SymbolKind::Concept: return "concept";
  case SymbolKind::CommentTag: return "comment-tag";
  }
  toolchain_unreachable("Garbage symbol kind");
}

void Symbol::print(raw_ostream &OS) const {
  OS << getName() << " | ";
  OS << getSymbolKindString(getSymbolKind()) << " | ";
  OS << getUSR();
}

void SymbolLocation::print(raw_ostream &OS) const {
  OS << getPath().getPathString() << ':' << getLine() << ':' << getColumn();
}

void SymbolOccurrence::foreachRelatedSymbol(SymbolRoleSet Roles,
                                            function_ref<void(SymbolRef)> Receiver) {
  for (auto &Rel : getRelations()) {
    if (Rel.getRoles().containsAny(Roles))
      Receiver(Rel.getSymbol());
  }
}

void SymbolOccurrence::print(raw_ostream &OS) const {
  getLocation().print(OS);
  OS << " | ";
  getSymbol()->print(OS);
  OS << " | ";
  printSymbolRoles(getRoles(), OS);
}


void IndexStoreDB::applyForEachSymbolRole(SymbolRoleSet Roles,
                                   toolchain::function_ref<void(SymbolRole)> Fn) {
#define APPLY_FOR_ROLE(Role) \
  if (Roles & SymbolRole::Role) \
    Fn(SymbolRole::Role)

  APPLY_FOR_ROLE(Declaration);
  APPLY_FOR_ROLE(Definition);
  APPLY_FOR_ROLE(Reference);
  APPLY_FOR_ROLE(Read);
  APPLY_FOR_ROLE(Write);
  APPLY_FOR_ROLE(Call);
  APPLY_FOR_ROLE(Dynamic);
  APPLY_FOR_ROLE(AddressOf);
  APPLY_FOR_ROLE(Implicit);
  APPLY_FOR_ROLE(RelationChildOf);
  APPLY_FOR_ROLE(RelationBaseOf);
  APPLY_FOR_ROLE(RelationOverrideOf);
  APPLY_FOR_ROLE(RelationReceivedBy);
  APPLY_FOR_ROLE(RelationCalledBy);
  APPLY_FOR_ROLE(RelationExtendedBy);
  APPLY_FOR_ROLE(RelationAccessorOf);
  APPLY_FOR_ROLE(RelationContainedBy);
  APPLY_FOR_ROLE(RelationIBTypeOf);
  APPLY_FOR_ROLE(RelationSpecializationOf);

#undef APPLY_FOR_ROLE
}

void IndexStoreDB::printSymbolRoles(SymbolRoleSet Roles, raw_ostream &OS) {
  bool VisitedOnce = false;
  applyForEachSymbolRole(Roles, [&](SymbolRole Role) {
    if (VisitedOnce)
      OS << ',';
    else
      VisitedOnce = true;
    switch (Role) {
    case SymbolRole::Declaration: OS << "Decl"; break;
    case SymbolRole::Definition: OS << "Def"; break;
    case SymbolRole::Reference: OS << "Ref"; break;
    case SymbolRole::Read: OS << "Read"; break;
    case SymbolRole::Write: OS << "Writ"; break;
    case SymbolRole::Call: OS << "Call"; break;
    case SymbolRole::Dynamic: OS << "Dyn"; break;
    case SymbolRole::AddressOf: OS << "Addr"; break;
    case SymbolRole::Implicit: OS << "Impl"; break;
    case SymbolRole::RelationChildOf: OS << "RelChild"; break;
    case SymbolRole::RelationBaseOf: OS << "RelBase"; break;
    case SymbolRole::RelationOverrideOf: OS << "RelOver"; break;
    case SymbolRole::RelationReceivedBy: OS << "RelRec"; break;
    case SymbolRole::RelationCalledBy: OS << "RelCall"; break;
    case SymbolRole::RelationExtendedBy: OS << "RelExt"; break;
    case SymbolRole::RelationAccessorOf: OS << "RelAcc"; break;
    case SymbolRole::RelationContainedBy: OS << "RelCont"; break;
    case SymbolRole::RelationIBTypeOf: OS << "RelIBType"; break;
    case SymbolRole::RelationSpecializationOf: OS << "RelSpecializationOf"; break;
    case SymbolRole::Canonical: OS << "Canon"; break;
    }
  });
}

Optional<SymbolProviderKind> IndexStoreDB::getSymbolProviderKindFromIdentifer(StringRef ident) {
  return toolchain::StringSwitch<Optional<SymbolProviderKind>>(ident)
    .Case("clang", SymbolProviderKind::Clang)
    .Case("language", SymbolProviderKind::Codira)
    .Default(toolchain::None);
}
