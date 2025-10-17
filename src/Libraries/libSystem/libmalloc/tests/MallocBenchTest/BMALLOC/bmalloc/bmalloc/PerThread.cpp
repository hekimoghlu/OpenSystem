/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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
#include "PerThread.h"

#include "BExport.h"
#include "Cache.h"
#include "Heap.h"

namespace bmalloc {

#if !HAVE_PTHREAD_MACHDEP_H

template<> BEXPORT bool PerThreadStorage<PerHeapKind<Cache>>::s_didInitialize = false;
template<> BEXPORT pthread_key_t PerThreadStorage<PerHeapKind<Cache>>::s_key = 0;
template<> BEXPORT std::once_flag PerThreadStorage<PerHeapKind<Cache>>::s_onceFlag = { };

template<> BEXPORT bool PerThreadStorage<PerHeapKind<Heap>>::s_didInitialize = false;
template<> BEXPORT pthread_key_t PerThreadStorage<PerHeapKind<Heap>>::s_key = 0;
template<> BEXPORT std::once_flag PerThreadStorage<PerHeapKind<Heap>>::s_onceFlag = { };

#endif

} // namespace bmalloc
