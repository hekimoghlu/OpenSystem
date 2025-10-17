/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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

//===- InMemoryStruct.h - Indirect Struct Access Smart Pointer --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_INMEMORYSTRUCT_H
#define LLVM_ADT_INMEMORYSTRUCT_H

#include <cassert>

namespace llvm {

/// \brief Helper object for abstracting access to an in-memory structure which
/// may require some kind of temporary storage.
///
/// This class is designed to be used for accessing file data structures which
/// in the common case can be accessed from a direct pointer to a memory mapped
/// object, but which in some cases may require indirect access to a temporary
/// structure (which, for example, may have undergone endianness translation).
template<typename T>
class InMemoryStruct {
  typedef T value_type;
  typedef value_type &reference;
  typedef value_type *pointer;
  typedef const value_type &const_reference;
  typedef const value_type *const_pointer;

  /// \brief The smart pointer target.
  value_type *Target;

  /// \brief A temporary object which can be used as a target of the smart
  /// pointer.
  value_type Contents;

private:

public:
  InMemoryStruct() : Target(0) {}
  InMemoryStruct(reference Value) : Target(&Contents), Contents(Value) {}
  InMemoryStruct(pointer Value) : Target(Value) {}
  InMemoryStruct(const InMemoryStruct<T> &Value) { *this = Value; }
  
  void operator=(const InMemoryStruct<T> &Value) {
    if (Value.Target != &Value.Contents) {
      Target = Value.Target;
    } else {
      Target = &Contents;
      Contents = Value.Contents;
    }
  }
  
  const_reference operator*() const {
    assert(Target && "Cannot dereference null pointer");
    return *Target;
  }
  reference operator*() {
    assert(Target && "Cannot dereference null pointer");
    return *Target;
  }

  const_pointer operator->() const {
    return Target;
  }
  pointer operator->() {
    return Target;
  }

  operator bool() const { return Target != 0; }
};

}

#endif
