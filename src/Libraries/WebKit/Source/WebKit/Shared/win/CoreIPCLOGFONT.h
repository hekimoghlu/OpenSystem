/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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

#if PLATFORM(WIN) && USE(CAIRO)

namespace WebKit {

class CoreIPCLOGFONT {
public:
    CoreIPCLOGFONT(const LOGFONT& data)
        : m_data(data)
    {
    }

    static LOGFONT create(std::span<const LOGFONT, 1> data)
    {
        return data.front();
    }

    std::span<const LOGFONT, 1> data() const
    {
        return std::span<const LOGFONT, 1> { &m_data, 1 };
    }

private:
    const LOGFONT& m_data;
};

} // namespace WebKit

#endif // PLATFORM(WIN) && USE(CAIRO)
