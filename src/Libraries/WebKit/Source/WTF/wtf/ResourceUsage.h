/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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

#include <array>
#include <wtf/Forward.h>

namespace WTF {

#if PLATFORM(COCOA)

struct TagInfo {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    size_t regionCount { 0 };
    size_t dirty { 0 };
    size_t reclaimable { 0 };
    size_t reserved { 0 };
};

WTF_EXPORT_PRIVATE ASCIILiteral displayNameForVMTag(unsigned);
WTF_EXPORT_PRIVATE size_t vmPageSize();
WTF_EXPORT_PRIVATE std::array<TagInfo, 256> pagesPerVMTag();
WTF_EXPORT_PRIVATE void logFootprintComparison(const std::array<TagInfo, 256>&, const std::array<TagInfo, 256>&);

#endif

}

#if PLATFORM(COCOA)
using WTF::TagInfo;
using WTF::displayNameForVMTag;
using WTF::vmPageSize;
using WTF::pagesPerVMTag;
#endif
