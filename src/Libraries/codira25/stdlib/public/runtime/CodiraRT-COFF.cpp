/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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

//===--- CodiraRT-COFF.cpp -------------------------------------------------===//
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

#include "ImageInspectionCommon.h"
#include "language/shims/MetadataSections.h"

#include <cstdint>
#include <new>

extern "C" const char __ImageBase[];

#define PASTE_EXPANDED(a,b) a##b
#define PASTE(a,b) PASTE_EXPANDED(a,b)

#define STRING_EXPANDED(string) #string
#define STRING(string) STRING_EXPANDED(string)

#define C_LABEL(name) PASTE(__USER_LABEL_PREFIX__,name)

#define PRAGMA(pragma) _Pragma(#pragma)

#define DECLARE_LANGUAGE_SECTION(name)                                            \
  PRAGMA(section("." #name "$A", long, read))                                  \
  __declspec(allocate("." #name "$A"))                                         \
  __declspec(align(1))                                                         \
  static uintptr_t __start_##name = 0;                                         \
                                                                               \
  PRAGMA(section("." #name "$C", long, read))                                  \
  __declspec(allocate("." #name "$C"))                                         \
  __declspec(align(1))                                                         \
  static uintptr_t __stop_##name = 0;

extern "C" {
DECLARE_LANGUAGE_SECTION(sw5prt)
DECLARE_LANGUAGE_SECTION(sw5prtc)
DECLARE_LANGUAGE_SECTION(sw5tymd)

DECLARE_LANGUAGE_SECTION(sw5tyrf)
DECLARE_LANGUAGE_SECTION(sw5rfst)
DECLARE_LANGUAGE_SECTION(sw5flmd)
DECLARE_LANGUAGE_SECTION(sw5asty)
DECLARE_LANGUAGE_SECTION(sw5repl)
DECLARE_LANGUAGE_SECTION(sw5reps)
DECLARE_LANGUAGE_SECTION(sw5bltn)
DECLARE_LANGUAGE_SECTION(sw5cptr)
DECLARE_LANGUAGE_SECTION(sw5mpen)
DECLARE_LANGUAGE_SECTION(sw5acfn)
DECLARE_LANGUAGE_SECTION(sw5ratt)
DECLARE_LANGUAGE_SECTION(sw5test)
}

namespace {
static language::MetadataSections sections{};
}

static void language_image_constructor() {
#define LANGUAGE_SECTION_RANGE(name)                                              \
  { reinterpret_cast<uintptr_t>(&__start_##name) + sizeof(__start_##name),     \
    reinterpret_cast<uintptr_t>(&__stop_##name) - reinterpret_cast<uintptr_t>(&__start_##name) - sizeof(__start_##name) }

  ::new (&sections) language::MetadataSections {
      language::CurrentSectionMetadataVersion,
      { __ImageBase },

      nullptr,
      nullptr,

      LANGUAGE_SECTION_RANGE(sw5prt),
      LANGUAGE_SECTION_RANGE(sw5prtc),
      LANGUAGE_SECTION_RANGE(sw5tymd),

      LANGUAGE_SECTION_RANGE(sw5tyrf),
      LANGUAGE_SECTION_RANGE(sw5rfst),
      LANGUAGE_SECTION_RANGE(sw5flmd),
      LANGUAGE_SECTION_RANGE(sw5asty),
      LANGUAGE_SECTION_RANGE(sw5repl),
      LANGUAGE_SECTION_RANGE(sw5reps),
      LANGUAGE_SECTION_RANGE(sw5bltn),
      LANGUAGE_SECTION_RANGE(sw5cptr),
      LANGUAGE_SECTION_RANGE(sw5mpen),
      LANGUAGE_SECTION_RANGE(sw5acfn),
      LANGUAGE_SECTION_RANGE(sw5ratt),
      LANGUAGE_SECTION_RANGE(sw5test),
  };

#undef LANGUAGE_SECTION_RANGE

  language_addNewDSOImage(&sections);
}

#pragma section(".CRT$XCIS", long, read)

__declspec(allocate(".CRT$XCIS"))
extern "C" void (*pCodiraImageConstructor)(void) = &language_image_constructor;
#pragma comment(linker, "/include:" STRING(C_LABEL(pCodiraImageConstructor)))

