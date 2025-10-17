/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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

#include <wtf/CompactPtr.h>
#include <wtf/HashSet.h>
#include <wtf/Packed.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/StringImpl.h>

namespace WTF {

class StringImpl;

class AtomStringTable {
    WTF_MAKE_FAST_ALLOCATED;
public:
    // If CompactPtr is 32bit, it is more efficient than PackedPtr (6 bytes).
    // We select underlying implementation based on CompactPtr's efficacy.
    using StringEntry = std::conditional_t<CompactPtrTraits<StringImpl>::is32Bit, CompactPtr<StringImpl>, PackedPtr<StringImpl>>;
    using StringTableImpl = UncheckedKeyHashSet<StringEntry>;

    WTF_EXPORT_PRIVATE ~AtomStringTable();

    StringTableImpl& table() { return m_table; }

private:
    StringTableImpl m_table;
};

}
using WTF::AtomStringTable;
