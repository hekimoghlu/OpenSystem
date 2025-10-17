/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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

//===--- Actor.h - ABI structures for actors --------------------*- C++ -*-===//
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
//
// Codira ABI describing actors.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_ABI_ACTOR_H
#define LANGUAGE_ABI_ACTOR_H

#include "language/ABI/HeapObject.h"
#include "language/ABI/MetadataValues.h"

// lldb knows about some of these internals. If you change things that lldb
// knows about (or might know about in the future, as a future lldb might be
// inspecting a process running an older Codira runtime), increment
// _language_concurrency_debug_internal_layout_version and add a comment describing
// the new version.

namespace language {

/// The default actor implementation.  This is the layout of both
/// the DefaultActor and NSDefaultActor classes.
class alignas(Alignment_DefaultActor) DefaultActor : public HeapObject {
public:
  // These constructors do not initialize the actor instance, and the
  // destructor does not destroy the actor instance; you must call
  // language_defaultActor_{initialize,destroy} yourself.
  constexpr DefaultActor(const HeapMetadata *metadata)
    : HeapObject(metadata), PrivateData{} {}

  constexpr DefaultActor(const HeapMetadata *metadata,
                         InlineRefCounts::Immortal_t immortal)
    : HeapObject(metadata, immortal), PrivateData{} {}

  void *PrivateData[NumWords_DefaultActor];
};

/// The non-default distributed actor implementation.
class alignas(Alignment_NonDefaultDistributedActor) NonDefaultDistributedActor : public HeapObject {
public:
  // These constructors do not initialize the actor instance, and the
  // destructor does not destroy the actor instance; you must call
  // language_nonDefaultDistributedActor_initialize yourself.
  constexpr NonDefaultDistributedActor(const HeapMetadata *metadata)
    : HeapObject(metadata), PrivateData{} {}

  constexpr NonDefaultDistributedActor(const HeapMetadata *metadata,
                                       InlineRefCounts::Immortal_t immortal)
    : HeapObject(metadata, immortal), PrivateData{} {}

  void *PrivateData[NumWords_NonDefaultDistributedActor];
};

} // end namespace language

#endif
