/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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

//===- Archive.h - ar archive file format -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ar archive file format class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ARCHIVE_H
#define LLVM_OBJECT_ARCHIVE_H

#include "llvm/Object/Binary.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
namespace object {

class Archive : public Binary {
  virtual void anchor();
public:
  class Child {
    const Archive *Parent;
    StringRef Data;

  public:
    Child(const Archive *p, StringRef d) : Parent(p), Data(d) {}

    bool operator ==(const Child &other) const {
      return (Parent == other.Parent) && (Data.begin() == other.Data.begin());
    }

    bool operator <(const Child &other) const {
      return Data.begin() < other.Data.begin();
    }

    Child getNext() const;
    error_code getName(StringRef &Result) const;
    int getLastModified() const;
    int getUID() const;
    int getGID() const;
    int getAccessMode() const;
    ///! Return the size of the archive member without the header or padding.
    uint64_t getSize() const;

    MemoryBuffer *getBuffer() const;
    error_code getAsBinary(OwningPtr<Binary> &Result) const;
  };

  class child_iterator {
    Child child;
  public:
    child_iterator() : child(Child(0, StringRef())) {}
    child_iterator(const Child &c) : child(c) {}
    const Child* operator->() const {
      return &child;
    }

    bool operator==(const child_iterator &other) const {
      return child == other.child;
    }

    bool operator!=(const child_iterator &other) const {
      return !(*this == other);
    }

    bool operator <(const child_iterator &other) const {
      return child < other.child;
    }

    child_iterator& operator++() {  // Preincrement
      child = child.getNext();
      return *this;
    }
  };

  class Symbol {
    const Archive *Parent;
    uint32_t SymbolIndex;
    uint32_t StringIndex; // Extra index to the string.

  public:
    bool operator ==(const Symbol &other) const {
      return (Parent == other.Parent) && (SymbolIndex == other.SymbolIndex);
    }

    Symbol(const Archive *p, uint32_t symi, uint32_t stri)
      : Parent(p)
      , SymbolIndex(symi)
      , StringIndex(stri) {}
    error_code getName(StringRef &Result) const;
    error_code getMember(child_iterator &Result) const;
    Symbol getNext() const;
  };

  class symbol_iterator {
    Symbol symbol;
  public:
    symbol_iterator(const Symbol &s) : symbol(s) {}
    const Symbol *operator->() const {
      return &symbol;
    }

    bool operator==(const symbol_iterator &other) const {
      return symbol == other.symbol;
    }

    bool operator!=(const symbol_iterator &other) const {
      return !(*this == other);
    }

    symbol_iterator& operator++() {  // Preincrement
      symbol = symbol.getNext();
      return *this;
    }
  };

  Archive(MemoryBuffer *source, error_code &ec);

  child_iterator begin_children(bool skip_internal = true) const;
  child_iterator end_children() const;

  symbol_iterator begin_symbols() const;
  symbol_iterator end_symbols() const;

  // Cast methods.
  static inline bool classof(Archive const *v) { return true; }
  static inline bool classof(Binary const *v) {
    return v->isArchive();
  }

private:
  child_iterator SymbolTable;
  child_iterator StringTable;
};

}
}

#endif
