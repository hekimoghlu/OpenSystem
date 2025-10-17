/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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

namespace WTF {
class PrintStream;
};

namespace JSC {

struct PrivateFieldPutKind {
    static constexpr uint8_t None = 0;
    static constexpr uint8_t Set = 1;
    static constexpr uint8_t Define = 2;

public:
    static constexpr PrivateFieldPutKind fromByte(uint8_t byte) { return PrivateFieldPutKind(byte); }
    static constexpr PrivateFieldPutKind none() { return PrivateFieldPutKind(None); }
    static constexpr PrivateFieldPutKind set() { return PrivateFieldPutKind(Set); }
    static constexpr PrivateFieldPutKind define() { return PrivateFieldPutKind(Define); }

    ALWAYS_INLINE bool isNone() const { return m_value == None; }
    ALWAYS_INLINE bool isSet() const { return m_value == Set; }
    ALWAYS_INLINE bool isDefine() const { return m_value == Define; }
    ALWAYS_INLINE uint8_t value() const { return m_value; }

    void dump(WTF::PrintStream&) const;

private:
    constexpr PrivateFieldPutKind(uint8_t value)
        : m_value(value)
    {
        ASSERT(m_value == None || m_value == Set || m_value == Define, m_value);
    }

    uint8_t m_value;
};

} // namespace JSC
