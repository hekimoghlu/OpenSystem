/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 3, 2022.
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

//===- Error.cpp - system_error extensions for Object -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines a new error_category for the Object library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Error.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace object;

namespace {
class _object_error_category : public _do_message {
public:
  virtual const char* name() const;
  virtual std::string message(int ev) const;
  virtual error_condition default_error_condition(int ev) const;
};
}

const char *_object_error_category::name() const {
  return "llvm.object";
}

std::string _object_error_category::message(int ev) const {
  switch (ev) {
  case object_error::success: return "Success";
  case object_error::invalid_file_type:
    return "The file was not recognized as a valid object file";
  case object_error::parse_failed:
    return "Invalid data was encountered while parsing the file";
  case object_error::unexpected_eof:
    return "The end of the file was unexpectedly encountered";
  default:
    llvm_unreachable("An enumerator of object_error does not have a message "
                     "defined.");
  }
}

error_condition _object_error_category::default_error_condition(int ev) const {
  if (ev == object_error::success)
    return errc::success;
  return errc::invalid_argument;
}

const error_category &object::object_category() {
  static _object_error_category o;
  return o;
}
