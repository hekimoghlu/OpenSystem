/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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

#include <libtasn1.h>
#include <optional>
#include <wtf/Forward.h>

namespace PAL {
namespace TASN1 {

class Structure {
public:
    Structure() = default;

    ~Structure()
    {
        asn1_delete_structure(&m_structure);
    }

    Structure(const Structure&) = delete;
    Structure& operator=(const Structure&) = delete;

    Structure(Structure&&) = delete;
    Structure& operator=(Structure&&) = delete;

    asn1_node* operator&() { return &m_structure; }
    operator asn1_node() const { return m_structure; }

private:
    asn1_node m_structure { nullptr };
};

bool createStructure(const char* elementName, asn1_node* root);
bool decodeStructure(asn1_node* root, const char* elementName, const Vector<uint8_t>& data);
std::optional<Vector<uint8_t>> elementData(asn1_node root, const char* elementName);
std::optional<Vector<uint8_t>> encodedData(asn1_node root, const char* elementName);
bool writeElement(asn1_node root, const char* elementName, const void* data, size_t dataSize);

} // namespace TASN1
} // namespace PAL
