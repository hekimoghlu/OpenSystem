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
#ifndef LIBWEBM_COMMON_INDENT_H_
#define LIBWEBM_COMMON_INDENT_H_

#include <string>

#include "mkvmuxer/mkvmuxertypes.h"

namespace libwebm {

const int kIncreaseIndent = 2;
const int kDecreaseIndent = -2;

// Used for formatting output so objects only have to keep track of spacing
// within their scope.
class Indent {
 public:
  explicit Indent(int indent);

  // Changes the number of spaces output. The value adjusted is relative to
  // |indent_|.
  void Adjust(int indent);

  std::string indent_str() const { return indent_str_; }

 private:
  // Called after |indent_| is changed. This will set |indent_str_| to the
  // proper number of spaces.
  void Update();

  int indent_;
  std::string indent_str_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(Indent);
};

}  // namespace libwebm

#endif  // LIBWEBM_COMMON_INDENT_H_
