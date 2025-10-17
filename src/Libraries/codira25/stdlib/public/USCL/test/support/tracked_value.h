/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SUPPORT_TRACKED_VALUE_H
#define SUPPORT_TRACKED_VALUE_H

#include <uscl/std/cassert>

#include "test_macros.h"

struct TrackedValue
{
  enum State
  {
    CONSTRUCTED,
    MOVED_FROM,
    DESTROYED
  };
  State state;

  TrackedValue()
      : state(State::CONSTRUCTED)
  {}

  TrackedValue(TrackedValue const& t)
      : state(State::CONSTRUCTED)
  {
    assert(t.state != State::MOVED_FROM && "copying a moved-from object");
    assert(t.state != State::DESTROYED && "copying a destroyed object");
  }

  TrackedValue(TrackedValue&& t)
      : state(State::CONSTRUCTED)
  {
    assert(t.state != State::MOVED_FROM && "double moving from an object");
    assert(t.state != State::DESTROYED && "moving from a destroyed object");
    t.state = State::MOVED_FROM;
  }

  TrackedValue& operator=(TrackedValue const& t)
  {
    assert(state != State::DESTROYED && "copy assigning into destroyed object");
    assert(t.state != State::MOVED_FROM && "copying a moved-from object");
    assert(t.state != State::DESTROYED && "copying a destroyed object");
    state = t.state;
    return *this;
  }

  TrackedValue& operator=(TrackedValue&& t)
  {
    assert(state != State::DESTROYED && "move assigning into destroyed object");
    assert(t.state != State::MOVED_FROM && "double moving from an object");
    assert(t.state != State::DESTROYED && "moving from a destroyed object");
    state   = t.state;
    t.state = State::MOVED_FROM;
    return *this;
  }

  ~TrackedValue()
  {
    assert(state != State::DESTROYED && "double-destroying an object");
    state = State::DESTROYED;
  }
};

#endif // SUPPORT_TRACKED_VALUE_H
