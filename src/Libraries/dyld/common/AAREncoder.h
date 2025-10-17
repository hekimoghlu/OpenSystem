/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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
#ifndef AAREncoder_h
#define AAREncoder_h

#include <span>
//#include <compression_private.h>

#include "Defines.h"
#include "Allocator.h"
#include "ByteStream.h"

struct VIS_HIDDEN AAREncoder {
    AAREncoder(lsl::Allocator& allocator);
    ~AAREncoder();
    void addFile(std::string_view path, std::span<std::byte> data);
    void addSymLink(std::string_view from, std::string_view to);
//    void setAlgorithm(compression_algorithm alg);
    void encode(ByteStream& output) const;
private:
    struct File {
        const char*             path;
        std::span<std::byte>    data;
    };
    struct Link {
        const char* from;
        const char* to;
    };
    uint16_t headerSize(const File& file) const;
    uint16_t headerSize(const Link& link) const;
    void encodeFile(const File& file, ByteStream& output) const;
    void encodeLink(const Link& link, ByteStream& output) const;
    lsl::Allocator*         _allocator  = nullptr;
//    compression_algorithm   _alg        = COMPRESSION_INVALID;
    lsl::Vector<File>       _files;
    lsl::Vector<Link>       _links;
};

#endif /* AAREncoder_h */
