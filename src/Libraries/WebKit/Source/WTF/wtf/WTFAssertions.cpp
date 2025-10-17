/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#include "config.h"

#include <wtf/Bag.h>
#include <wtf/Platform.h>
#include <wtf/RefPtr.h>

#if OS(DARWIN)
#include <mach/vm_param.h>
#include <mach/vm_types.h>
#endif

namespace WTF {

namespace {
struct DummyStruct { };
}

static_assert(sizeof(Bag<DummyStruct>) == sizeof(void*));

static_assert(sizeof(Ref<DummyStruct>) == sizeof(DummyStruct*));

static_assert(sizeof(RefPtr<DummyStruct>) == sizeof(DummyStruct*));

#if OS(DARWIN) && CPU(ADDRESS64)
// NaN boxing encoding relies on this.
static_assert(MACH_VM_MAX_ADDRESS <= (1ull << 48));
#endif

} // namespace WTF

