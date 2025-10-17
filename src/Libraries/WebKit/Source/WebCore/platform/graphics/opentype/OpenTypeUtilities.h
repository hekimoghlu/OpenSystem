/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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
#ifndef OpenTypeUtilities_h
#define OpenTypeUtilities_h

#include <windows.h>
#include <wtf/Forward.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct BigEndianUShort;
class SharedBuffer;
struct EOTPrefix;
class FontMemoryResource;
class FragmentedSharedBuffer;

struct EOTHeader {
    EOTHeader();

    size_t size() const { return m_buffer.size(); }
    const uint8_t* data() const { return m_buffer.data(); }

    EOTPrefix* prefix() { return reinterpret_cast<EOTPrefix*>(m_buffer.data()); }
    void updateEOTSize(size_t);
    void appendBigEndianString(const BigEndianUShort*, unsigned short length);
    void appendPaddingShort();

private:
    Vector<uint8_t, 512> m_buffer;
};

bool renameFont(const SharedBuffer&, const String&, Vector<uint8_t>&);
RefPtr<FontMemoryResource> renameAndActivateFont(const SharedBuffer&, const String&);

} // namespace WebCore

#endif // OpenTypeUtilities_h
