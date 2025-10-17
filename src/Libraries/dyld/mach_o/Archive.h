/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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
#ifndef mach_o_Archive_h
#define mach_o_Archive_h

#include <TargetConditionals.h>

#if !TARGET_OS_EXCLAVEKIT

// Darwin
#include <ar.h>

// stl
#include <string_view>
#include <optional>

// mach_o
#include "Header.h"

namespace mach_o {

class Entry : ar_hdr
{
public:
    void                        getName(char *, int) const;
    uint64_t                    modificationTime() const;
    Error                       content(std::span<const uint8_t>& content) const;
    Error                       next(Entry*& next) const;
    Error                       valid() const;

    static uint64_t             extendedFormatNameSize(std::string_view name);
    static uint64_t             entrySize(bool extendedFormatNames, std::string_view name, uint64_t contentSize);
    static size_t               write(std::span<uint8_t> buffer, bool extendedFormatNames, std::string_view name, uint64_t mktime, std::span<const uint8_t> content);

private:
    bool                        hasLongName() const;
    uint64_t                    getLongNameSpace() const;
};

// if a member file in a static library has this name, then force load it
#define ALWAYS_LOAD_MEMBER_NAME "__ALWAYS_LOAD.o"

/*!
 * @class Archive
 *
 * @abstract
 *      Abstraction for static archives
 */
class VIS_HIDDEN Archive
{
public:

    struct Member
    {
        std::string_view            name;
        std::span<const uint8_t>    contents;
        uint64_t                    mtime;
        unsigned                    memberIndex;
    };

    static std::optional<Archive>   isArchive(std::span<const uint8_t> buffer);

    mach_o::Error   forEachMember(void (^handler)(const Member&, bool& stop)) const;
    mach_o::Error   forEachMachO(void (^handler)(const Member&, const mach_o::Header*, bool& stop)) const;

    std::span<const uint8_t> buffer;

    constexpr static std::string_view archive_magic = "!<arch>\n";

protected:

    Archive(std::span<const uint8_t> buffer): buffer(buffer) {}
};
}

#endif // !TARGET_OS_EXCLAVEKIT

#endif // mach_o_Archive_h
