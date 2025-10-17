/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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

//===--- ASTDumper.h - Codira AST Dumper flags -------------------*- C++ -*-===//
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
// This file defines types that are used to control the level of detail printed
// by the AST dumper.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_AST_AST_DUMPER_H
#define LANGUAGE_AST_AST_DUMPER_H

namespace language {

/// Describes the nature of requests that should be kicked off, if any, to
/// compute members and top-level decls when dumping an AST.
enum class ASTDumpMemberLoading {
  /// Dump cached members if available; if they are not, do not kick off any
  /// parsing or type-checking requests.
  None,

  /// Dump parsed members, kicking off a parsing request if necessary to compute
  /// them, but not performing additional type-checking.
  Parsed,

  /// Dump all fully-type checked members, kicking off any requests necessary to
  /// compute them.
  TypeChecked,
};

} // namespace language

#endif // LANGUAGE_AST_AST_DUMPER_H