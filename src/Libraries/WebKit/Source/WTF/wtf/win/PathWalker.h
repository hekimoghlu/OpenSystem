/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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

#include <Windows.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>

namespace WTF {

class PathWalker {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_MAKE_NONCOPYABLE(PathWalker);
public:
    PathWalker(const WTF::String& directory, const WTF::String& pattern);
    ~PathWalker();

    bool isValid() const { return m_handle != INVALID_HANDLE_VALUE; }
    const WIN32_FIND_DATAW& data() const { return m_data; }

    bool step();

private:
    HANDLE m_handle;
    WIN32_FIND_DATAW m_data;
};

} // namespace WTF
