/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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

//===- CXXFieldCollector.h - Utility class for C++ class semantic analysis ===//
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
//
//  This file provides CXXFieldCollector that is used during parsing & semantic
//  analysis of C++ classes.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_CORE_SEMA_CXXFIELDCOLLECTOR_H
#define LANGUAGE_CORE_SEMA_CXXFIELDCOLLECTOR_H

#include "language/Core/Basic/LLVM.h"
#include "toolchain/ADT/SmallVector.h"

namespace language::Core {
  class FieldDecl;

/// CXXFieldCollector - Used to keep track of CXXFieldDecls during parsing of
/// C++ classes.
class CXXFieldCollector {
  /// Fields - Contains all FieldDecls collected during parsing of a C++
  /// class. When a nested class is entered, its fields are appended to the
  /// fields of its parent class, when it is exited its fields are removed.
  SmallVector<FieldDecl*, 32> Fields;

  /// FieldCount - Each entry represents the number of fields collected during
  /// the parsing of a C++ class. When a nested class is entered, a new field
  /// count is pushed, when it is exited, the field count is popped.
  SmallVector<size_t, 4> FieldCount;

  // Example:
  //
  // class C {
  //   int x,y;
  //   class NC {
  //     int q;
  //     // At this point, Fields contains [x,y,q] decls and FieldCount contains
  //     // [2,1].
  //   };
  //   int z;
  //   // At this point, Fields contains [x,y,z] decls and FieldCount contains
  //   // [3].
  // };

public:
  /// StartClass - Called by Sema::ActOnStartCXXClassDef.
  void StartClass() { FieldCount.push_back(0); }

  /// Add - Called by Sema::ActOnCXXMemberDeclarator.
  void Add(FieldDecl *D) {
    Fields.push_back(D);
    ++FieldCount.back();
  }

  /// getCurNumField - The number of fields added to the currently parsed class.
  size_t getCurNumFields() const {
    assert(!FieldCount.empty() && "no currently-parsed class");
    return FieldCount.back();
  }

  /// getCurFields - Pointer to array of fields added to the currently parsed
  /// class.
  FieldDecl **getCurFields() { return &*(Fields.end() - getCurNumFields()); }

  /// FinishClass - Called by Sema::ActOnFinishCXXClassDef.
  void FinishClass() {
    Fields.resize(Fields.size() - getCurNumFields());
    FieldCount.pop_back();
  }
};

} // end namespace language::Core

#endif
