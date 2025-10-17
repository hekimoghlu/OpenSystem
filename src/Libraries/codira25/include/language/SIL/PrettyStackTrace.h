/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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

//===--- PrettyStackTrace.h - Crash trace information -----------*- C++ -*-===//
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
// This file defines SIL-specific RAII classes that give better diagnostic
// output about when, exactly, a crash is occurring.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_SIL_PRETTYSTACKTRACE_H
#define LANGUAGE_SIL_PRETTYSTACKTRACE_H

#include "language/SIL/SILLocation.h"
#include "language/SIL/SILNode.h"
#include "language/SIL/SILDeclRef.h"
#include "toolchain/ADT/SmallString.h"
#include "toolchain/ADT/Twine.h"
#include "toolchain/Support/PrettyStackTrace.h"

namespace language {
class ASTContext;
class SILFunction;

void printSILLocationDescription(toolchain::raw_ostream &out, SILLocation loc,
                                 ASTContext &ctx);

/// PrettyStackTraceLocation - Observe that we are doing some
/// processing starting at a SIL location.
class PrettyStackTraceSILLocation : public toolchain::PrettyStackTraceEntry {
  SILLocation Loc;
  const char *Action;
  ASTContext &Context;
public:
  PrettyStackTraceSILLocation(const char *action, SILLocation loc,
                              ASTContext &C)
    : Loc(loc), Action(action), Context(C) {}
  virtual void print(toolchain::raw_ostream &OS) const override;
};


/// Observe that we are doing some processing of a SIL function.
class PrettyStackTraceSILFunction : public toolchain::PrettyStackTraceEntry {
  const SILFunction *fn;

  /// An inline buffer of characters used if we are passed a twine.
  SmallString<256> data;

  /// This points either at a user provided const char * string or points at the
  /// inline message buffer that is initialized with data from a twine on
  /// construction.
  StringRef action;

public:
  PrettyStackTraceSILFunction(const char *action, const SILFunction *fn)
      : fn(fn), data(), action(action) {}

  PrettyStackTraceSILFunction(toolchain::Twine &&twine, const SILFunction *fn)
      : fn(fn), data(), action(twine.toNullTerminatedStringRef(data)) {}

  virtual void print(toolchain::raw_ostream &os) const override;

protected:
  void printFunctionInfo(toolchain::raw_ostream &out) const;
};

/// Observe that we are visiting SIL nodes.
class PrettyStackTraceSILNode : public toolchain::PrettyStackTraceEntry {
  const SILNode *Node;
  const char *Action;

public:
  PrettyStackTraceSILNode(const char *action, SILNodePointer node)
    : Node(node), Action(action) {}

  virtual void print(toolchain::raw_ostream &OS) const override;
};

/// Observe that we are processing a reference to a SIL decl.
class PrettyStackTraceSILDeclRef : public toolchain::PrettyStackTraceEntry {
  SILDeclRef declRef;
  StringRef action;

public:
  PrettyStackTraceSILDeclRef(const char *action, SILDeclRef declRef)
      : declRef(declRef), action(action) {}

  virtual void print(toolchain::raw_ostream &os) const override;
};

} // end namespace language

#endif
