/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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
#include "BytecodeCacheError.h"

#include "JSGlobalObject.h"
#include <wtf/SafeStrerror.h>
#include <wtf/text/MakeString.h>

namespace JSC {

bool BytecodeCacheError::StandardError::isValid() const
{
    return true;
}

String BytecodeCacheError::StandardError::message() const
{
    return String::fromLatin1(safeStrerror(m_errno).data());
}

bool BytecodeCacheError::WriteError::isValid() const
{
    return true;
}

String BytecodeCacheError::WriteError::message() const
{
    return makeString("Could not write the full cache file to disk. Only wrote "_s, m_written, " of the expected "_s, m_expected, " bytes."_s);
}

BytecodeCacheError& BytecodeCacheError::operator=(const ParserError& error)
{
    m_error = error;
    return *this;
}

BytecodeCacheError& BytecodeCacheError::operator=(const StandardError& error)
{
    m_error = error;
    return *this;
}

BytecodeCacheError& BytecodeCacheError::operator=(const WriteError& error)
{
    m_error = error;
    return *this;
}

bool BytecodeCacheError::isValid() const
{
    return WTF::switchOn(m_error, [](const auto& error) {
        return error.isValid();
    });
}

String BytecodeCacheError::message() const
{
    return WTF::switchOn(m_error, [](const auto& error) {
        return error.message();
    });
}

} // namespace JSC
