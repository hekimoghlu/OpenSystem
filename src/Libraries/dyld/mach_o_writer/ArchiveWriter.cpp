/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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
// mach_o_writer
#include "ArchiveWriter.h"

// stl
#include <string_view>

// Darwin
#include <ar.h>
#include <mach-o/ranlib.h>
#include <mach/mach.h>
#include <mach/vm_map.h>

namespace mach_o {

size_t ArchiveWriter::size(std::span<const Member> members, bool extendedFormatNames)
{
    uint64_t size = archive_magic.size();

    for ( const Member& m : members ) {
        size += Entry::entrySize(extendedFormatNames, m.name, m.contents.size());
    }

    return size;
}

Error ArchiveWriter::make(std::span<uint8_t> buffer, std::span<const Member> members, bool extendedFormatNames)
{
    if ( buffer.size() < archive_magic.size() ) {
        return Error("buffer to small");
    }

    std::span<uint8_t> remainingSpace = buffer;
    memcpy(remainingSpace.data(), archive_magic.data(), archive_magic.size());
    remainingSpace = remainingSpace.subspan(archive_magic.size());

    for ( const Member& m : members ) {
        size_t writtenBytes = Entry::write(remainingSpace, extendedFormatNames, m.name, m.mtime, m.contents);
        if ( writtenBytes > remainingSpace.size() ) {
            assert(false && "invalid buffer size");
            return Error("buffer to small");
        }

        remainingSpace = remainingSpace.subspan(writtenBytes);
    }

    assert(remainingSpace.empty());
    if ( isArchive(buffer).has_value() )
        return Error::none();
    return Error("error writing archive");
}

} // namespace mach_o
