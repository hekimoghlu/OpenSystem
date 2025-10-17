/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 27, 2025.
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
#ifndef mach_o_writer_Archive_h
#define mach_o_writer_Archive_h

// stl
#include <string_view>
#include <optional>
#include <vector>

// mach_o
#include "Archive.h"
#include "Header.h"

namespace mach_o {

using namespace mach_o;

/*!
 * @class ArchiveWriter
 *
 * @abstract
 *      Abstraction for building static archives
 */
class VIS_HIDDEN ArchiveWriter : public Archive
{
public:
    // for building
    static size_t   size(std::span<const Member> members, bool extendedFormatNames = true);
    static Error    make(std::span<uint8_t> buffer, std::span<const Member> members, bool extendedFormatNames = true);

private:

    ArchiveWriter(std::span<const uint8_t> buffer) : Archive(buffer) {}
};
}

#endif // mach_o_writer_Archive_h
