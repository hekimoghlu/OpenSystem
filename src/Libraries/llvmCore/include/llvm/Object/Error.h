/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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

//===- Error.h - system_error extensions for Object -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This declares a new error_category for the Object library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ERROR_H
#define LLVM_OBJECT_ERROR_H

#include "llvm/Support/system_error.h"

namespace llvm {
namespace object {

const error_category &object_category();

struct object_error {
enum _ {
  success = 0,
  invalid_file_type,
  parse_failed,
  unexpected_eof
};
  _ v_;

  object_error(_ v) : v_(v) {}
  explicit object_error(int v) : v_(_(v)) {}
  operator int() const {return v_;}
};

inline error_code make_error_code(object_error e) {
  return error_code(static_cast<int>(e), object_category());
}

} // end namespace object.

template <> struct is_error_code_enum<object::object_error> : true_type { };

template <> struct is_error_code_enum<object::object_error::_> : true_type { };

} // end namespace llvm.

#endif
