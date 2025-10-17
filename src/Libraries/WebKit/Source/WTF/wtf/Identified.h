/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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

#include <wtf/UUID.h>

namespace WTF {

template <typename IdentifierType>
class IdentifiedBase {
public:
    IdentifierType identifier() const { return m_identifier; }

protected:
    IdentifiedBase(const IdentifiedBase&) = default;

    explicit IdentifiedBase(IdentifierType identifier)
        : m_identifier(identifier)
    {
    }

    IdentifiedBase& operator=(const IdentifiedBase&) = default;

private:
    IdentifierType m_identifier;
};

template <typename ObjectIdentifierType>
class Identified : public IdentifiedBase<ObjectIdentifierType> {
protected:
    Identified()
        : IdentifiedBase<ObjectIdentifierType>(ObjectIdentifierType::generate())
    {
    }

    Identified(const Identified&) = default;
    Identified& operator=(const Identified&) = default;
};

template <typename T>
class UUIDIdentified : public IdentifiedBase<UUID> {
protected:
    UUIDIdentified()
        : IdentifiedBase<UUID>(UUID::createVersion4())
    {
    }

    UUIDIdentified(const UUIDIdentified&) = default;
};

} // namespace WTF

using WTF::Identified;
using WTF::UUIDIdentified;
