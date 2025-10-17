/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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
#include "config.h"
#include "IDBIndexInfo.h"

#include <wtf/CrossThreadCopier.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

IDBIndexInfo::IDBIndexInfo(IDBIndexIdentifier identifier, IDBObjectStoreIdentifier objectStoreIdentifier, const String& name, IDBKeyPath&& keyPath, bool unique, bool multiEntry)
    : m_identifier(identifier)
    , m_objectStoreIdentifier(objectStoreIdentifier)
    , m_name(name)
    , m_keyPath(WTFMove(keyPath))
    , m_unique(unique)
    , m_multiEntry(multiEntry)
{
}

IDBIndexInfo IDBIndexInfo::isolatedCopy() const &
{
    return { m_identifier, m_objectStoreIdentifier, m_name.isolatedCopy(), crossThreadCopy(m_keyPath), m_unique, m_multiEntry };
}

IDBIndexInfo IDBIndexInfo::isolatedCopy() &&
{
    return { m_identifier, m_objectStoreIdentifier, WTFMove(m_name).isolatedCopy(), crossThreadCopy(WTFMove(m_keyPath)), m_unique, m_multiEntry };
}

#if !LOG_DISABLED

String IDBIndexInfo::loggingString(int indent) const
{
    StringBuilder indentString;
    for (int i = 0; i < indent; ++i)
        indentString.append(' ');
    return makeString(indentString.toString(), "Index: "_s, m_name, " ("_s, m_identifier, ") keyPath: "_s, WebCore::loggingString(m_keyPath), '\n');
}

String IDBIndexInfo::condensedLoggingString() const
{
    return makeString("<Idx: "_s, m_name, " ("_s, m_identifier, "), OS ("_s, m_objectStoreIdentifier, ")>"_s);
}

#endif

} // namespace WebCore
