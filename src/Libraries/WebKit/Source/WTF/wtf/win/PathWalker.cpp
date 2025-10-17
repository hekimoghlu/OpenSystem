/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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
#include <wtf/win/PathWalker.h>

#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WTF {

PathWalker::PathWalker(const String& directory, const String& pattern)
{
    String path = makeString(directory, '\\', pattern);
    m_handle = ::FindFirstFileW(path.wideCharacters().data(), &m_data);
}

PathWalker::~PathWalker()
{
    if (!isValid())
        return;
    ::FindClose(m_handle);
}

bool PathWalker::step()
{
    return ::FindNextFileW(m_handle, &m_data);
}

} // namespace WTF
