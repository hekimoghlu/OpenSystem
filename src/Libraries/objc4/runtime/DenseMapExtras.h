/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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
#ifndef DENSEMAPEXTRAS_H
#define DENSEMAPEXTRAS_H

#include "InitWrappers.h"
#include "llvm-DenseMap.h"
#include "llvm-DenseSet.h"

namespace objc {

// Convenience class for Dense Maps & Sets
template <typename Key, typename Value>
class ExplicitInitDenseMap : public ExplicitInit<DenseMap<Key, Value>> { };

template <typename Key, typename Value>
class LazyInitDenseMap : public LazyInit<DenseMap<Key, Value>> { };

template <typename Value>
class ExplicitInitDenseSet : public ExplicitInit<DenseSet<Value>> { };

template <typename Value>
class LazyInitDenseSet : public LazyInit<DenseSet<Value>> { };

} // namespace objc

#endif /* DENSEMAPEXTRAS_H */
