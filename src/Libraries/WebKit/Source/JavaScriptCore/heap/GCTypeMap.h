/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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

#include "CollectionScope.h"
#include <wtf/Assertions.h>

namespace JSC {

template<typename T>
struct GCTypeMap {
    T eden;
    T full;
    
    T& operator[](CollectionScope scope)
    {
        switch (scope) {
        case CollectionScope::Full:
            return full;
        case CollectionScope::Eden:
            return eden;
        }
        ASSERT_NOT_REACHED();
        return full;
    }
    
    const T& operator[](CollectionScope scope) const
    {
        switch (scope) {
        case CollectionScope::Full:
            return full;
        case CollectionScope::Eden:
            return eden;
        }
        RELEASE_ASSERT_NOT_REACHED();
        return full;
    }
};

} // namespace JSC

