/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 12, 2024.
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

//===--- NodeDumper.cpp - Codira Demangling Debug Dump Functions -----------===//
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

#include "language/Demangling/Demangle.h"
#include "language/Demangling/Demangler.h"
#include <cstdio>

using namespace language;
using namespace Demangle;

const char *Demangle::getNodeKindString(language::Demangle::Node::Kind k) {
  switch (k) {
#define NODE(ID)                                                               \
  case Node::Kind::ID:                                                         \
    return #ID;
#include "language/Demangling/DemangleNodes.def"
  }
  return "Demangle::Node::Kind::???";
}

static void printNode(DemanglerPrinter &Out, const Node *node, unsigned depth) {
  // Indent two spaces per depth.
  for (unsigned i = 0; i < depth * 2; ++i) {
    Out << ' ';
  }
  if (!node) {
    Out << "<<NULL>>";
    return;
  }
  Out << "kind=" << getNodeKindString(node->getKind());
  if (node->hasText()) {
    Out << ", text=\"" << node->getText() << '\"';
  }
  if (node->hasIndex()) {
    Out << ", index=" << node->getIndex();
  }
  Out << '\n';
  for (auto &child : *node) {
    printNode(Out, child, depth + 1);
  }
}

std::string Demangle::getNodeTreeAsString(NodePointer Root) {
  DemanglerPrinter Printer;
  printNode(Printer, Root, 0);
  return std::move(Printer).str();
}

void language::Demangle::Node::dump() {
  std::string TreeStr = getNodeTreeAsString(this);
  fputs(TreeStr.c_str(), stderr);
}

void Demangler::dump() {
  for (unsigned Idx = 0; Idx < Substitutions.size(); ++Idx) {
    fprintf(stderr, "Substitution[%c]:\n", Idx + 'A');
    Substitutions[Idx]->dump();
    fprintf(stderr, "\n");
  }

  for (unsigned Idx = 0; Idx < NodeStack.size(); ++Idx) {
    fprintf(stderr, "NodeStack[%u]:\n", Idx);
    NodeStack[Idx]->dump();
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "Position = %zd:\n%.*s\n%*s\n", Pos,
          (int)Text.size(), Text.data(), (int)Pos + 1, "^");
}

