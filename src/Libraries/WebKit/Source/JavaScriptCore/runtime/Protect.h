/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 8, 2023.
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
#pragma once

#include "Heap.h"
#include "JSCJSValue.h"

namespace JSC {

inline void gcProtect(JSCell* val) 
{
    Heap::heap(val)->protect(val);
}

inline void gcUnprotect(JSCell* val)
{
    Heap::heap(val)->unprotect(val);
}

inline void gcProtectNullTolerant(JSCell* val) 
{
    if (val) 
        gcProtect(val);
}

inline void gcUnprotectNullTolerant(JSCell* val) 
{
    if (val) 
        gcUnprotect(val);
}

inline void gcProtect(JSValue value)
{
    if (value && value.isCell())
        gcProtect(value.asCell());
}

inline void gcUnprotect(JSValue value)
{
    if (value && value.isCell())
        gcUnprotect(value.asCell());
}

} // namespace JSC
