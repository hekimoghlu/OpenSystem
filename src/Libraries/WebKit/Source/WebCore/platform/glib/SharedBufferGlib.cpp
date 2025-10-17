/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
#include "SharedBuffer.h"

#include <wtf/glib/GUniquePtr.h>
#include <wtf/text/CString.h>

#include <glib.h>

namespace WebCore {

FragmentedSharedBuffer::FragmentedSharedBuffer(GBytes* bytes)
{
    ASSERT(bytes);
    m_size = g_bytes_get_size(bytes);
    m_segments.append({ 0, DataSegment::create(GRefPtr<GBytes>(bytes)) });
}

Ref<FragmentedSharedBuffer> FragmentedSharedBuffer::create(GBytes* bytes)
{
    return adoptRef(*new FragmentedSharedBuffer(bytes));
}

GRefPtr<GBytes> SharedBuffer::createGBytes() const
{
    ref();
    GRefPtr<GBytes> bytes = adoptGRef(g_bytes_new_with_free_func(span().data(), size(), [](gpointer data) {
        static_cast<SharedBuffer*>(data)->deref();
    }, const_cast<SharedBuffer*>(this)));
    return bytes;
}

} // namespace WebCore
