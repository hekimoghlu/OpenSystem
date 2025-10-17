/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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

//===---------------------- system_error.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This was lifted from libc++ and modified for C++03.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/system_error.h"
#include "llvm/Support/Errno.h"
#include <string>
#include <cstring>

namespace llvm {

// class error_category

error_category::error_category() {
}

error_category::~error_category() {
}

error_condition
error_category::default_error_condition(int ev) const {
  return error_condition(ev, *this);
}

bool
error_category::equivalent(int code, const error_condition& condition) const {
  return default_error_condition(code) == condition;
}

bool
error_category::equivalent(const error_code& code, int condition) const {
  return *this == code.category() && code.value() == condition;
}

std::string
_do_message::message(int ev) const {
  return std::string(sys::StrError(ev));
}

class _generic_error_category : public _do_message {
public:
  virtual const char* name() const LLVM_OVERRIDE;
  virtual std::string message(int ev) const LLVM_OVERRIDE;
};

const char*
_generic_error_category::name() const {
  return "generic";
}

std::string
_generic_error_category::message(int ev) const {
#ifdef ELAST
  if (ev > ELAST)
    return std::string("unspecified generic_category error");
#endif  // ELAST
  return _do_message::message(ev);
}

const error_category&
generic_category() {
  static _generic_error_category s;
  return s;
}

class _system_error_category : public _do_message {
public:
  virtual const char* name() const LLVM_OVERRIDE;
  virtual std::string message(int ev) const LLVM_OVERRIDE;
  virtual error_condition default_error_condition(int ev) const LLVM_OVERRIDE;
};

const char*
_system_error_category::name() const {
  return "system";
}

// std::string _system_error_category::message(int ev) const {
// Is in Platform/system_error.inc

// error_condition _system_error_category::default_error_condition(int ev) const
// Is in Platform/system_error.inc

const error_category&
system_category() {
  static _system_error_category s;
  return s;
}

const error_category&
posix_category() {
#ifdef LLVM_ON_WIN32
  return generic_category();
#else
  return system_category();
#endif
}

// error_condition

std::string
error_condition::message() const {
  return _cat_->message(_val_);
}

// error_code

std::string
error_code::message() const {
  return _cat_->message(_val_);
}

} // end namespace llvm

// Include the truly platform-specific parts of this class.
#if defined(LLVM_ON_UNIX)
#include "Unix/system_error.inc"
#endif
#if defined(LLVM_ON_WIN32)
#include "Windows/system_error.inc"
#endif
