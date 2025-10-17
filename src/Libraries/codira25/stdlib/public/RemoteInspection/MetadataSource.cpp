/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 23, 2024.
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

//===--- MetadataSource.cpp - Codira Metadata Sources for Reflection -------===//
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

#if LANGUAGE_ENABLE_REFLECTION

#include "language/RemoteInspection/MetadataSource.h"
#include <iostream>

using namespace language;
using namespace reflection;

class PrintMetadataSource
  : public MetadataSourceVisitor<PrintMetadataSource, void> {
  std::ostream &stream;
  unsigned Indent;

  std::ostream &indent(unsigned Amount) {
    for (unsigned i = 0; i < Amount; ++i)
      stream << " ";
    return stream;
  }

  std::ostream &printHeader(std::string Name) {
    indent(Indent) << "(" << Name;
    return stream;
  }

  std::ostream &printField(std::string name, std::string value) {
    if (!name.empty())
      stream << " " << name << "=" << value;
    else
      stream << " " << value;
    return stream;
  }

  void printRec(const MetadataSource *MS) {
    stream << "\n";

    Indent += 2;
    visit(MS);
    Indent -= 2;
  }

  void closeForm() {
    stream << ")";
  }

public:
  PrintMetadataSource(std::ostream &stream, unsigned Indent)
      : stream(stream), Indent(Indent) {}

  void
  visitClosureBindingMetadataSource(const ClosureBindingMetadataSource *CB) {
    printHeader("closure_binding");
    printField("index", std::to_string(CB->getIndex()));
    closeForm();
  }

  void
  visitReferenceCaptureMetadataSource(const ReferenceCaptureMetadataSource *RC){
    printHeader("reference_capture");
    printField("index", std::to_string(RC->getIndex()));
    closeForm();
  }

  void
  visitMetadataCaptureMetadataSource(const MetadataCaptureMetadataSource *MC){
    printHeader("metadata_capture");
    printField("index", std::to_string(MC->getIndex()));
    closeForm();
  }

  void
  visitGenericArgumentMetadataSource(const GenericArgumentMetadataSource *GA) {
    printHeader("generic_argument");
    printField("index", std::to_string(GA->getIndex()));
    printRec(GA->getSource());
    closeForm();
  }

  void visitSelfMetadataSource(const SelfMetadataSource *S) {
    printHeader("this");
    closeForm();
  }

  void
  visitSelfWitnessTableMetadataSource(const SelfWitnessTableMetadataSource *W) {
    printHeader("self_witness_table");
    closeForm();
  }
};

void MetadataSource::dump() const { dump(std::cerr, 0); }

void MetadataSource::dump(std::ostream &stream, unsigned Indent) const {
  PrintMetadataSource(stream, Indent).visit(this);
  stream << "\n";
}

#endif
