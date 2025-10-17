/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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

//===--- CFTypeInfo.cpp - Information about CF types  ---------------------===//
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
// This file provides support for reasoning about CF types
//
//===----------------------------------------------------------------------===//

#include "CFTypeInfo.h"
#include "ImporterImpl.h"

using namespace language;
using namespace importer;

namespace {
  // Quasi-lexicographic order: string length first, then string data.
  // Since we don't care about the actual length, we can use this, which
  // lets us ignore the string data a larger proportion of the time.
  struct SortByLengthComparator {
    bool operator()(StringRef lhs, StringRef rhs) const {
      return (lhs.size() < rhs.size() ||
              (lhs.size() == rhs.size() && lhs < rhs));
    }
  };
} // end anonymous namespace

/// The list of known CF types.  We use 'constexpr' to verify that this is
/// emitted as a constant.  Note that this is expected to be sorted in
/// quasi-lexicographic order.
static constexpr const toolchain::StringLiteral KnownCFTypes[] = {
#define CF_TYPE(NAME) #NAME,
#define NON_CF_TYPE(NAME)
#include "SortedCFDatabase.def"
};
const size_t NumKnownCFTypes = sizeof(KnownCFTypes) / sizeof(*KnownCFTypes);

/// Maintain a set of known CF types.
bool CFPointeeInfo::isKnownCFTypeName(StringRef name) {
  return std::binary_search(KnownCFTypes, KnownCFTypes + NumKnownCFTypes,
                            name, SortByLengthComparator());
}

/// Classify a potential CF typedef.
CFPointeeInfo
CFPointeeInfo::classifyTypedef(const language::Core::TypedefNameDecl *typedefDecl) {
  language::Core::QualType type = typedefDecl->getUnderlyingType();

  if (auto elaborated = type->getAs<language::Core::ElaboratedType>())
    type = elaborated->desugar();

  if (auto subTypedef = type->getAs<language::Core::TypedefType>()) {
    if (classifyTypedef(subTypedef->getDecl()))
      return forTypedef(subTypedef->getDecl());
    return forInvalid();
  }

  if (auto ptr = type->getAs<language::Core::PointerType>()) {
    auto pointee = ptr->getPointeeType();

    // Must be 'const' or nothing.
    language::Core::Qualifiers quals = pointee.getQualifiers();
    bool isConst = quals.hasConst();
    quals.removeConst();
    if (quals.empty()) {
      if (auto record = pointee->getAs<language::Core::RecordType>()) {
        auto recordDecl = record->getDecl();
        if (recordDecl->hasAttr<language::Core::ObjCBridgeAttr>() ||
            recordDecl->hasAttr<language::Core::ObjCBridgeMutableAttr>() ||
            recordDecl->hasAttr<language::Core::ObjCBridgeRelatedAttr>() ||
            isKnownCFTypeName(typedefDecl->getName())) {
          return forRecord(isConst, record->getDecl());
        }
      } else if (pointee->isVoidType()) {
        if (typedefDecl->hasAttr<language::Core::ObjCBridgeAttr>() ||
            isKnownCFTypeName(typedefDecl->getName())) {
          return isConst ? forConstVoid() : forVoid();
        }
      }
    }
  }

  return forInvalid();
}

bool importer::isCFTypeDecl(
       const language::Core::TypedefNameDecl *Decl) {
  if (CFPointeeInfo::classifyTypedef(Decl))
    return true;
  return false;
}

StringRef importer::getCFTypeName(
            const language::Core::TypedefNameDecl *decl) {
  if (auto pointee = CFPointeeInfo::classifyTypedef(decl)) {
    auto name = decl->getName();
    if (pointee.isRecord() || pointee.isTypedef())
      if (name.ends_with(LANGUAGE_CFTYPE_SUFFIX))
        return name.drop_back(strlen(LANGUAGE_CFTYPE_SUFFIX));

    return name;
  }

  return "";
}
