/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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

#include <wtf/text/SymbolImpl.h>
#include <wtf/text/WTFString.h>

namespace JSC {

class PrivateName {
public:
    explicit PrivateName(SymbolImpl& uid)
        : m_uid(uid)
    {
    }

    enum DescriptionTag { Description };
    explicit PrivateName(DescriptionTag, const String& description)
        : m_uid(SymbolImpl::create(*description.impl()))
    {
    }

    enum PrivateSymbolTag { PrivateSymbol };
    explicit PrivateName(PrivateSymbolTag, const String& description)
        : m_uid(PrivateSymbolImpl::create(*description.impl()))
    {
    }

    PrivateName(const PrivateName& privateName)
        : m_uid(privateName.m_uid.copyRef())
    {
    }

    PrivateName(PrivateName&&) = default;

    SymbolImpl& uid() const { return m_uid; }

    bool operator==(const PrivateName& other) const { return &uid() == &other.uid(); }

private:
    Ref<SymbolImpl> m_uid;
};

} // namespace JSC
