/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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

#include "ISOBox.h"

namespace WebCore {

class WEBCORE_EXPORT ISOProtectionSystemSpecificHeaderBox : public ISOFullBox {
public:
    using KeyID = Vector<uint8_t>;

    ISOProtectionSystemSpecificHeaderBox();
    ~ISOProtectionSystemSpecificHeaderBox();

    static FourCC boxTypeName() { return std::span { "pssh" }; }

    static std::optional<Vector<uint8_t>> peekSystemID(JSC::DataView&, unsigned offset);

    Vector<uint8_t> systemID() const { return m_systemID; }
    Vector<KeyID> keyIDs() const { return m_keyIDs; }
    Vector<uint8_t> data() const { return m_data; }

    bool parse(JSC::DataView&, unsigned& offset) override;

protected:
    Vector<uint8_t> m_systemID;
    Vector<KeyID> m_keyIDs;
    Vector<uint8_t> m_data;
};

}

SPECIALIZE_TYPE_TRAITS_ISOBOX(ISOProtectionSystemSpecificHeaderBox)
