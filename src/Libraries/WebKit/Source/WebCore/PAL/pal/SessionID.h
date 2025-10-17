/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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

#include <optional>
#include <wtf/HashFunctions.h>
#include <wtf/HashTraits.h>

namespace PAL {

class SessionID {
public:
    SessionID() = delete;

    enum SessionConstants : uint64_t {
        EphemeralSessionMask    = 0x8000000000000000,
        DefaultSessionID        = 1,
        LegacyPrivateSessionID  = DefaultSessionID | EphemeralSessionMask,
        HashTableEmptyValueID   = 0,
        HashTableDeletedValueID = std::numeric_limits<uint64_t>::max(),
    };

    static SessionID defaultSessionID() { return SessionID(DefaultSessionID); }
    static SessionID legacyPrivateSessionID() { return SessionID(LegacyPrivateSessionID); }

    explicit SessionID(WTF::HashTableDeletedValueType)
        : m_identifier(HashTableDeletedValueID)
    {
    }

    explicit SessionID(WTF::HashTableEmptyValueType)
        : m_identifier(HashTableEmptyValueID)
    {
    }

    explicit SessionID(uint64_t identifier)
        : m_identifier(identifier)
    {
    }

    PAL_EXPORT static SessionID generateEphemeralSessionID();
    PAL_EXPORT static SessionID generatePersistentSessionID();
    PAL_EXPORT static void enableGenerationProtection();

    bool isValid() const { return isValidSessionIDValue(m_identifier); }
    bool isEphemeral() const { return m_identifier & EphemeralSessionMask && m_identifier != HashTableDeletedValueID; }
    bool isHashTableDeletedValue() const { return m_identifier == HashTableDeletedValueID; }
    bool isHashTableEmptyValue() const { return m_identifier == HashTableEmptyValueID; }

    uint64_t toUInt64() const { return m_identifier; }
    friend bool operator==(SessionID, SessionID) = default;
    bool isAlwaysOnLoggingAllowed() const { return !isEphemeral(); }

    SessionID isolatedCopy() const { return *this; }

    explicit operator bool() const { return m_identifier; }

    static bool isValidSessionIDValue(uint64_t sessionID) { return sessionID != HashTableEmptyValueID && sessionID != HashTableDeletedValueID; }
private:
    uint64_t m_identifier;
};

} // namespace PAL

namespace WTF {

struct SessionIDHash {
    static unsigned hash(const PAL::SessionID& p) { return intHash(p.toUInt64()); }
    static bool equal(const PAL::SessionID& a, const PAL::SessionID& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

template<> struct HashTraits<PAL::SessionID> : GenericHashTraits<PAL::SessionID> {
    static PAL::SessionID emptyValue() { return PAL::SessionID(HashTableEmptyValue); }
    static bool isEmptyValue(const PAL::SessionID& value) { return value.isHashTableEmptyValue(); }
    static void constructDeletedValue(PAL::SessionID& slot) { new (NotNull, &slot) PAL::SessionID(HashTableDeletedValue); }
    static bool isDeletedValue(const PAL::SessionID& slot) { return slot.isHashTableDeletedValue(); }
};

template<> struct DefaultHash<PAL::SessionID> : SessionIDHash { };

} // namespace WTF
