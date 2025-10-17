/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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

#include <wtf/ASCIICType.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPUAdapterInfo : public RefCounted<GPUAdapterInfo> {
public:
    static Ref<GPUAdapterInfo> create(String&& name)
    {
        return adoptRef(*new GPUAdapterInfo(WTFMove(name)));
    }

    String vendor() const { auto v = m_name.split(' '); return v.size() ? normalizedIdentifier(v[0]) : ""_s; }
    String architecture() const { return vendor(); }
    String device() const { return vendor(); }
    String description() const { return vendor(); }

private:
    GPUAdapterInfo(String&& name)
        : m_name(name)
    {
    }
    static String normalizedIdentifier(const String& s) { return s.convertToLowercaseWithoutLocale().removeCharacters([](auto c) { return !isASCIIAlphanumeric(c); }); }

    String m_name;
};

}
