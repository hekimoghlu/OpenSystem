/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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

#if USE(CG)

namespace WebCore {

class CGContextStateSaver {
public:
    CGContextStateSaver(CGContextRef context, bool saveAndRestore = true)
        : m_context(context)
        , m_saveAndRestore(saveAndRestore)
    {
        if (m_saveAndRestore)
            CGContextSaveGState(m_context);
    }
    
    ~CGContextStateSaver()
    {
        if (m_saveAndRestore)
            CGContextRestoreGState(m_context);
    }
    
    void save()
    {
        ASSERT(!m_saveAndRestore);
        CGContextSaveGState(m_context);
        m_saveAndRestore = true;
    }

    void restore()
    {
        ASSERT(m_saveAndRestore);
        CGContextRestoreGState(m_context);
        m_saveAndRestore = false;
    }
    
    bool didSave() const
    {
        return m_saveAndRestore;
    }
    
private:
    CGContextRef m_context;
    bool m_saveAndRestore;
};

} // namespace WebCore

#endif // USE(CG)
