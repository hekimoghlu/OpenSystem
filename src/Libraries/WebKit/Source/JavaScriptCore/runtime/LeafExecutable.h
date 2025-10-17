/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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

#include <wtf/HashMap.h>

namespace JSC {

class LeafExecutable;
class UnlinkedFunctionExecutable;

using LeafExecutableMap = UncheckedKeyHashMap<const UnlinkedFunctionExecutable*, LeafExecutable>;

class LeafExecutable {
public:
    LeafExecutable() = default;

    LeafExecutable(ptrdiff_t offset)
        : m_base(offset)
    {
    }

    ptrdiff_t base() const { return m_base; }
    LeafExecutable operator+(size_t) const;

private:
    ptrdiff_t m_base;
};

} // namespace JSC
