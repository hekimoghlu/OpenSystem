/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
#include "FunctionAllowlist.h"

#if ENABLE(JIT)

#include "CodeBlock.h"
#include <stdio.h>
#include <string.h>
#include <wtf/SafeStrerror.h>
#include <wtf/text/MakeString.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

FunctionAllowlist::FunctionAllowlist(const char* filename)
{
    if (!filename)
        return;

    FILE* f = fopen(filename, "r");
    if (!f) {
        if (errno == ENOENT) {
            m_hasActiveAllowlist = true;
            m_entries.add(String::fromLatin1(filename));
        } else
            dataLogF("Failed to open file %s. Did you add the file-read-data entitlement to WebProcess.sb? Error code: %s\n", filename, safeStrerror(errno).data());
        return;
    }

    m_hasActiveAllowlist = true;

    char* line;
    char buffer[BUFSIZ];
    while ((line = fgets(buffer, sizeof(buffer), f))) {
        if (strstr(line, "//") == line)
            continue;

        // Get rid of newlines at the ends of the strings.
        size_t length = strlen(line);
        if (length && line[length - 1] == '\n') {
            line[length - 1] = '\0';
            length--;
        }

        // Skip empty lines.
        if (!length)
            continue;
        
        m_entries.add(String(unsafeMakeSpan(line, length)));
    }

    int result = fclose(f);
    if (result)
        dataLogF("Failed to close file %s: %s\n", filename, safeStrerror(errno).data());
}

bool FunctionAllowlist::contains(CodeBlock* codeBlock) const
{
    if (!m_hasActiveAllowlist)
        return true;

    if (m_entries.isEmpty())
        return false;

    String name = String::fromUTF8(codeBlock->inferredName().span());
    if (m_entries.contains(name))
        return true;

    String hash = String::fromUTF8(codeBlock->hashAsStringIfPossible().span());
    if (m_entries.contains(hash))
        return true;

    return m_entries.contains(makeString(name, '#', hash));
}

bool FunctionAllowlist::shouldDumpWasmFunction(uint32_t index) const
{
    if (!m_hasActiveAllowlist)
        return false;
    return containsWasmFunction(index);
}

bool FunctionAllowlist::containsWasmFunction(uint32_t index) const
{
    if (!m_hasActiveAllowlist)
        return true;

    if (m_entries.isEmpty())
        return false;
    return m_entries.contains(String::number(index));
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(JIT)
