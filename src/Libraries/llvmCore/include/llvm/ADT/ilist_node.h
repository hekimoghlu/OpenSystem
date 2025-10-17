/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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

//==-- llvm/ADT/ilist_node.h - Intrusive Linked List Helper ------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ilist_node class template, which is a convenient
// base class for creating classes that can be used with ilists.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ILIST_NODE_H
#define LLVM_ADT_ILIST_NODE_H

namespace llvm {

template<typename NodeTy>
struct ilist_traits;

/// ilist_half_node - Base class that provides prev services for sentinels.
///
template<typename NodeTy>
class ilist_half_node {
  friend struct ilist_traits<NodeTy>;
  NodeTy *Prev;
protected:
  NodeTy *getPrev() { return Prev; }
  const NodeTy *getPrev() const { return Prev; }
  void setPrev(NodeTy *P) { Prev = P; }
  ilist_half_node() : Prev(0) {}
};

template<typename NodeTy>
struct ilist_nextprev_traits;

/// ilist_node - Base class that provides next/prev services for nodes
/// that use ilist_nextprev_traits or ilist_default_traits.
///
template<typename NodeTy>
class ilist_node : private ilist_half_node<NodeTy> {
  friend struct ilist_nextprev_traits<NodeTy>;
  friend struct ilist_traits<NodeTy>;
  NodeTy *Next;
  NodeTy *getNext() { return Next; }
  const NodeTy *getNext() const { return Next; }
  void setNext(NodeTy *N) { Next = N; }
protected:
  ilist_node() : Next(0) {}

public:
  /// @name Adjacent Node Accessors
  /// @{

  /// \brief Get the previous node, or 0 for the list head.
  NodeTy *getPrevNode() {
    NodeTy *Prev = this->getPrev();

    // Check for sentinel.
    if (!Prev->getNext())
      return 0;

    return Prev;
  }

  /// \brief Get the previous node, or 0 for the list head.
  const NodeTy *getPrevNode() const {
    const NodeTy *Prev = this->getPrev();

    // Check for sentinel.
    if (!Prev->getNext())
      return 0;

    return Prev;
  }

  /// \brief Get the next node, or 0 for the list tail.
  NodeTy *getNextNode() {
    NodeTy *Next = getNext();

    // Check for sentinel.
    if (!Next->getNext())
      return 0;

    return Next;
  }

  /// \brief Get the next node, or 0 for the list tail.
  const NodeTy *getNextNode() const {
    const NodeTy *Next = getNext();

    // Check for sentinel.
    if (!Next->getNext())
      return 0;

    return Next;
  }

  /// @}
};

} // End llvm namespace

#endif
