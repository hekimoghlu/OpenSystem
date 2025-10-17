/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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

//===--- SerializedDiagnosticConsumer.h - Serialize Diagnostics -*- C++ -*-===//
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
//  This file defines the SerializedDiagnosticConsumer class, which
//  serializes diagnostics to Clang's serialized diagnostic bitcode format.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_SERIALIZEDDIAGNOSTICCONSUMER_H
#define LANGUAGE_SERIALIZEDDIAGNOSTICCONSUMER_H

#include <memory>

namespace toolchain {
  class StringRef;
}

namespace language {

  class DiagnosticConsumer;

  namespace serialized_diagnostics {
    /// Create a DiagnosticConsumer that serializes diagnostics to a file, using
    /// Clang's serialized diagnostics format.
    ///
    /// \param outputPath the file path to write the diagnostics to.
    ///
    /// \returns A new diagnostic consumer that serializes diagnostics.
    std::unique_ptr<DiagnosticConsumer>
    createConsumer(toolchain::StringRef outputPath, bool emitMacroExpansionFiles);
  }
}

#endif
