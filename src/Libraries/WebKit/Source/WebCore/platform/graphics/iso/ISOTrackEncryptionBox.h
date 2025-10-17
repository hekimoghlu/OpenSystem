/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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

class WEBCORE_EXPORT ISOTrackEncryptionBox final : public ISOFullBox {
public:
    ISOTrackEncryptionBox();
    ~ISOTrackEncryptionBox();

    static FourCC boxTypeName() { return std::span { "tenc" }; }

    std::optional<int8_t> defaultCryptByteBlock() const { return m_defaultCryptByteBlock; }
    std::optional<int8_t> defaultSkipByteBlock() const { return m_defaultSkipByteBlock; }
    int8_t defaultIsProtected() const { return m_defaultIsProtected; }
    int8_t defaultPerSampleIVSize() const { return m_defaultPerSampleIVSize; }
    Vector<uint8_t> defaultKID() const { return m_defaultKID; }
    Vector<uint8_t> defaultConstantIV() const { return m_defaultConstantIV; }

    bool parseWithoutTypeAndSize(JSC::DataView&);

    bool parse(JSC::DataView&, unsigned& offset) override;

private:
    bool parsePayload(JSC::DataView&, unsigned& offset);

    std::optional<int8_t> m_defaultCryptByteBlock;
    std::optional<int8_t> m_defaultSkipByteBlock;
    int8_t m_defaultIsProtected { 0 };
    int8_t m_defaultPerSampleIVSize { 0 };
    Vector<uint8_t> m_defaultKID;
    Vector<uint8_t> m_defaultConstantIV;
};

}

SPECIALIZE_TYPE_TRAITS_ISOBOX(ISOTrackEncryptionBox)
