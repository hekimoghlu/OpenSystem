/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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

#include "LineMetadata.h"
#include "Mutex.h"
#include "Sizes.h"
#include "StaticPerProcess.h"
#include "Vector.h"
#include <array>
#include <mutex>

#if !BUSE(LIBPAS)

namespace bmalloc {

class HeapConstants : public StaticPerProcess<HeapConstants> {
public:
    HeapConstants(const LockHolder&);
    ~HeapConstants() = delete;

    inline size_t pageClass(size_t sizeClass) const { return m_pageClasses[sizeClass]; }
    inline size_t smallLineCount() const { return bmalloc::smallLineCount(m_vmPageSizePhysical); }
    inline unsigned char startOffset(size_t sizeClass, size_t lineNumber) const { return lineMetadata(sizeClass, lineNumber).startOffset; }
    inline unsigned char objectCount(size_t sizeClass, size_t lineNumber) const { return lineMetadata(sizeClass, lineNumber).objectCount; }

private:
    void initializeLineMetadata();
    void initializePageMetadata();

    inline const LineMetadata& lineMetadata(size_t sizeClass, size_t lineNumber) const
    {
        return m_smallLineMetadata[sizeClass * smallLineCount() + lineNumber];
    }

    size_t m_vmPageSizePhysical;
    const LineMetadata* m_smallLineMetadata { };
    Vector<LineMetadata> m_smallLineMetadataStorage;
    std::array<size_t, sizeClassCount> m_pageClasses;
};
BALLOW_DEPRECATED_DECLARATIONS_BEGIN
DECLARE_STATIC_PER_PROCESS_STORAGE(HeapConstants);
BALLOW_DEPRECATED_DECLARATIONS_END

} // namespace bmalloc

#endif
