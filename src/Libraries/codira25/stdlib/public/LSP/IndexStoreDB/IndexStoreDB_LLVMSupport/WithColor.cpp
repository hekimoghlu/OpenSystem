/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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

//===- WithColor.cpp ------------------------------------------------------===//
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

#include <IndexStoreDB_LLVMSupport/toolchain_Support_WithColor.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_raw_ostream.h>

using namespace toolchain;

cl::OptionCategory toolchain::ColorCategory("Color Options");

static cl::opt<cl::boolOrDefault>
    UseColor("color", cl::cat(ColorCategory),
             cl::desc("Use colors in output (default=autodetect)"),
             cl::init(cl::BOU_UNSET));

WithColor::WithColor(raw_ostream &OS, HighlightColor Color, bool DisableColors)
    : OS(OS), DisableColors(DisableColors) {
  // Detect color from terminal type unless the user passed the --color option.
  if (colorsEnabled()) {
    switch (Color) {
    case HighlightColor::Address:
      OS.changeColor(raw_ostream::YELLOW);
      break;
    case HighlightColor::String:
      OS.changeColor(raw_ostream::GREEN);
      break;
    case HighlightColor::Tag:
      OS.changeColor(raw_ostream::BLUE);
      break;
    case HighlightColor::Attribute:
      OS.changeColor(raw_ostream::CYAN);
      break;
    case HighlightColor::Enumerator:
      OS.changeColor(raw_ostream::MAGENTA);
      break;
    case HighlightColor::Macro:
      OS.changeColor(raw_ostream::RED);
      break;
    case HighlightColor::Error:
      OS.changeColor(raw_ostream::RED, true);
      break;
    case HighlightColor::Warning:
      OS.changeColor(raw_ostream::MAGENTA, true);
      break;
    case HighlightColor::Note:
      OS.changeColor(raw_ostream::BLACK, true);
      break;
    case HighlightColor::Remark:
      OS.changeColor(raw_ostream::BLUE, true);
      break;
    }
  }
}

raw_ostream &WithColor::error() { return error(errs()); }

raw_ostream &WithColor::warning() { return warning(errs()); }

raw_ostream &WithColor::note() { return note(errs()); }

raw_ostream &WithColor::remark() { return remark(errs()); }

raw_ostream &WithColor::error(raw_ostream &OS, StringRef Prefix,
                              bool DisableColors) {
  if (!Prefix.empty())
    OS << Prefix << ": ";
  return WithColor(OS, HighlightColor::Error, DisableColors).get()
         << "error: ";
}

raw_ostream &WithColor::warning(raw_ostream &OS, StringRef Prefix,
                                bool DisableColors) {
  if (!Prefix.empty())
    OS << Prefix << ": ";
  return WithColor(OS, HighlightColor::Warning, DisableColors).get()
         << "warning: ";
}

raw_ostream &WithColor::note(raw_ostream &OS, StringRef Prefix,
                             bool DisableColors) {
  if (!Prefix.empty())
    OS << Prefix << ": ";
  return WithColor(OS, HighlightColor::Note, DisableColors).get() << "note: ";
}

raw_ostream &WithColor::remark(raw_ostream &OS, StringRef Prefix,
                               bool DisableColors) {
  if (!Prefix.empty())
    OS << Prefix << ": ";
  return WithColor(OS, HighlightColor::Remark, DisableColors).get()
         << "remark: ";
}

bool WithColor::colorsEnabled() {
  if (DisableColors)
    return false;
  if (UseColor == cl::BOU_UNSET)
    return OS.has_colors();
  return UseColor == cl::BOU_TRUE;
}

WithColor &WithColor::changeColor(raw_ostream::Colors Color, bool Bold,
                                  bool BG) {
  if (colorsEnabled())
    OS.changeColor(Color, Bold, BG);
  return *this;
}

WithColor &WithColor::resetColor() {
  if (colorsEnabled())
    OS.resetColor();
  return *this;
}

WithColor::~WithColor() { resetColor(); }
