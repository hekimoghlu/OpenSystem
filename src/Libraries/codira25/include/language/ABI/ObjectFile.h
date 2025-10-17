/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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

//===--- ObjectFile.h - Object File Related Information ------*- C++ -*-===//
//
// Object File related data structures.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_ABI_OBJECTFILE_H
#define LANGUAGE_ABI_OBJECTFILE_H

#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/ErrorHandling.h"
#include <optional>

namespace language {

/// Represents the nine reflection sections used by Codira + the Codira AST
/// section used by the debugger.
enum ReflectionSectionKind : uint8_t {
#define HANDLE_LANGUAGE_SECTION(KIND, MACHO, ELF, COFF) KIND,
#include "toolchain/BinaryFormat/Codira.def"
#undef HANDLE_LANGUAGE_SECTION
};

/// Abstract base class responsible for providing the correct reflection section
/// string identifier for a given object file type (Mach-O, ELF, COFF).
class CodiraObjectFileFormat {
public:
  virtual ~CodiraObjectFileFormat() {}
  virtual toolchain::StringRef getSectionName(ReflectionSectionKind section) = 0;
  virtual std::optional<toolchain::StringRef> getSegmentName() { return {}; }
  /// Get the name of the segment in the symbol rich binary that may contain
  /// Codira metadata.
  virtual std::optional<toolchain::StringRef> getSymbolRichSegmentName() {
    return {};
  }
  /// Predicate to identify if the named section can contain reflection data.
  virtual bool sectionContainsReflectionData(toolchain::StringRef sectionName) = 0;
};

/// Responsible for providing the Mach-O reflection section identifiers.
class CodiraObjectFileFormatMachO : public CodiraObjectFileFormat {
public:
  toolchain::StringRef getSectionName(ReflectionSectionKind section) override {
    switch (section) {
#define HANDLE_LANGUAGE_SECTION(KIND, MACHO, ELF, COFF)                           \
  case KIND:                                                                   \
    return MACHO;
#include "toolchain/BinaryFormat/Codira.def"
#undef HANDLE_LANGUAGE_SECTION
    }
    toolchain_unreachable("Section type not found.");
  }

  std::optional<toolchain::StringRef> getSegmentName() override {
    return {"__TEXT"};
  }

  std::optional<toolchain::StringRef> getSymbolRichSegmentName() override {
    return {"__DWARF"};
  }

  bool sectionContainsReflectionData(toolchain::StringRef sectionName) override {
    return sectionName.starts_with("__language5_") || sectionName == "__const";
  }
};

/// Responsible for providing the ELF reflection section identifiers.
class CodiraObjectFileFormatELF : public CodiraObjectFileFormat {
public:
  toolchain::StringRef getSectionName(ReflectionSectionKind section) override {
    switch (section) {
#define HANDLE_LANGUAGE_SECTION(KIND, MACHO, ELF, COFF)                           \
  case KIND:                                                                   \
    return ELF;
#include "toolchain/BinaryFormat/Codira.def"
#undef HANDLE_LANGUAGE_SECTION
    }
    toolchain_unreachable("Section type not found.");
  }

  bool sectionContainsReflectionData(toolchain::StringRef sectionName) override {
    return sectionName.starts_with("language5_");
  }
};

/// Responsible for providing the COFF reflection section identifiers
class CodiraObjectFileFormatCOFF : public CodiraObjectFileFormat {
public:
  toolchain::StringRef getSectionName(ReflectionSectionKind section) override {
    switch (section) {
#define HANDLE_LANGUAGE_SECTION(KIND, MACHO, ELF, COFF)                           \
  case KIND:                                                                   \
    return COFF;
#include "toolchain/BinaryFormat/Codira.def"
#undef HANDLE_LANGUAGE_SECTION
    }
    toolchain_unreachable("Section  not found.");
  }

  bool sectionContainsReflectionData(toolchain::StringRef sectionName) override {
    return sectionName.starts_with(".sw5");
  }
};
} // namespace language
#endif // LANGUAGE_ABI_OBJECTFILE_H
