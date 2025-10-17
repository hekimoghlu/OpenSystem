/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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

//===--- ClangDerivedConformances.h -----------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_CLANG_DERIVED_CONFORMANCES_H
#define LANGUAGE_CLANG_DERIVED_CONFORMANCES_H

#include "ImporterImpl.h"
#include "language/AST/ASTContext.h"

namespace language {

bool isIterator(const language::Core::CXXRecordDecl *clangDecl);

bool isUnsafeStdMethod(const language::Core::CXXMethodDecl *methodDecl);

/// If the decl is a C++ input iterator, synthesize a conformance to the
/// UnsafeCxxInputIterator protocol, which is defined in the Cxx module.
void conformToCxxIteratorIfNeeded(ClangImporter::Implementation &impl,
                                  NominalTypeDecl *decl,
                                  const language::Core::CXXRecordDecl *clangDecl);

/// If the decl defines `operator bool()`, synthesize a conformance to the
/// CxxConvertibleToBool protocol, which is defined in the Cxx module.
void conformToCxxConvertibleToBoolIfNeeded(
    ClangImporter::Implementation &impl, NominalTypeDecl *decl,
    const language::Core::CXXRecordDecl *clangDecl);

/// If the decl is an instantiation of C++ `std::optional`, synthesize a
/// conformance to CxxOptional protocol, which is defined in the Cxx module.
void conformToCxxOptionalIfNeeded(ClangImporter::Implementation &impl,
                                  NominalTypeDecl *decl,
                                  const language::Core::CXXRecordDecl *clangDecl);

/// If the decl is a C++ sequence, synthesize a conformance to the CxxSequence
/// protocol, which is defined in the Cxx module.
void conformToCxxSequenceIfNeeded(ClangImporter::Implementation &impl,
                                  NominalTypeDecl *decl,
                                  const language::Core::CXXRecordDecl *clangDecl);

/// If the decl is an instantiation of C++ `std::set`, `std::unordered_set` or
/// `std::multiset`, synthesize a conformance to CxxSet, which is defined in the
/// Cxx module.
void conformToCxxSetIfNeeded(ClangImporter::Implementation &impl,
                             NominalTypeDecl *decl,
                             const language::Core::CXXRecordDecl *clangDecl);

/// If the decl is an instantiation of C++ `std::pair`, synthesize a conformance
/// to CxxPair, which is defined in the Cxx module.
void conformToCxxPairIfNeeded(ClangImporter::Implementation &impl,
                              NominalTypeDecl *decl,
                              const language::Core::CXXRecordDecl *clangDecl);

/// If the decl is an instantiation of C++ `std::map` or `std::unordered_map`,
/// synthesize a conformance to CxxDictionary, which is defined in the Cxx module.
void conformToCxxDictionaryIfNeeded(ClangImporter::Implementation &impl,
                                    NominalTypeDecl *decl,
                                    const language::Core::CXXRecordDecl *clangDecl);

/// If the decl is an instantiation of C++ `std::vector`, synthesize a
/// conformance to CxxVector, which is defined in the Cxx module.
void conformToCxxVectorIfNeeded(ClangImporter::Implementation &impl,
                                NominalTypeDecl *decl,
                                const language::Core::CXXRecordDecl *clangDecl);

/// If the decl is an instantiation of C++ `std::function`, synthesize a
/// conformance to CxxFunction, which is defined in the Cxx module.
void conformToCxxFunctionIfNeeded(ClangImporter::Implementation &impl,
                                  NominalTypeDecl *decl,
                                  const language::Core::CXXRecordDecl *clangDecl);
                                  
/// If the decl is an instantiation of C++ `std::span`, synthesize a
/// conformance to CxxSpan, which is defined in the Cxx module.
void conformToCxxSpanIfNeeded(ClangImporter::Implementation &impl,
                                NominalTypeDecl *decl,
                                const language::Core::CXXRecordDecl *clangDecl);

} // namespace language

#endif // LANGUAGE_CLANG_DERIVED_CONFORMANCES_H
