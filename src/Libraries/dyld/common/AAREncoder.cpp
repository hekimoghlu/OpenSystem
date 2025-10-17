/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
#include "Cksum.h"
#include "AAREncoder.h"

using lsl::Allocator;

#define PARALLEL_COMPRESS_BLOCK_SIZE (0x4000000)

// Figure out how large of integer is needed to store the value
// Figure out how large of integer is needed to store the value
static uint8_t byteSizeForValue(size_t value) {
    // Check to see if value fits by generating an inverse mask of 2^result-1 and see if any bits leak
    for(uint8_t i = 1; i < 8; i<<=1) {
        if ((value & ~((1ULL<<(i*8))-1)) == 0) {
            return i;
        }
    }
    return 8;
}

uint16_t AAREncoder::headerSize(const File& file) const {
    size_t headerSize = 0;
    return headerSize;
}

void AAREncoder::encodeFile(const File& file, ByteStream& output) const {
    uint32_t checksum = cksum(file.data);
    output.insert(output.end(), file.data.begin(), file.data.end());
}

uint16_t AAREncoder::headerSize(const Link& link) const {
    size_t headerSize = 0;
    return headerSize;
}

void AAREncoder::encodeLink(const Link& link, ByteStream& output) const {
}

AAREncoder::AAREncoder(lsl::Allocator& allocator) : _allocator(&allocator), _files(allocator), _links(allocator) {}
AAREncoder::~AAREncoder() {
    for (auto& file : _files) {
        _allocator->free((void*)file.path);
    }
    for (auto& link : _links) {
        _allocator->free((void*)link.to);
        _allocator->free((void*)link.from);
    }
}
void AAREncoder::addFile(std::string_view path, std::span<std::byte> data) {
    char* pathStr = (char*)_allocator->malloc(path.size()+1);
    memcpy(pathStr, path.data(), path.size());
    pathStr[path.size()] = 0;
    _files.push_back({pathStr, data});
}
void AAREncoder::addSymLink(std::string_view from, std::string_view to) {
    char* fromStr   = (char*)_allocator->malloc(from.size()+1);
    char* toStr     = (char*)_allocator->malloc(to.size()+1);
    memcpy(fromStr, from.data(), from.size());
    fromStr[from.size()] = 0;
    memcpy(toStr, to.data(), to.size());
    toStr[to.size()] = 0;
    _links.push_back({ fromStr, toStr });
}

void AAREncoder::encode(ByteStream& output) const {
    ByteStream fileStream = ByteStream(*_allocator);

    for (auto link : _links) {
        encodeLink(link, fileStream);
    }
    for (auto file : _files) {
        encodeFile(file, fileStream);
    }
    output.insert(output.end(), fileStream.begin(), fileStream.end());
}
