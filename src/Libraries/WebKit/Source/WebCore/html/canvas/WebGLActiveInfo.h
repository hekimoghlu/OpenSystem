/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 9, 2022.
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

#if ENABLE(WEBGL)

#include "GraphicsTypesGL.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class WebGLActiveInfo : public RefCounted<WebGLActiveInfo> {
public:
    static Ref<WebGLActiveInfo> create(const String& name, GCGLenum type, GCGLint size)
    {
        return adoptRef(*new WebGLActiveInfo(name, type, size));
    }
    String name() const { return m_name; }
    GCGLenum type() const { return m_type; }
    GCGLint size() const { return m_size; }

private:
    WebGLActiveInfo(const String& name, GCGLenum type, GCGLint size)
        : m_name(name)
        , m_type(type)
        , m_size(size)
    {
        ASSERT(name.length());
        ASSERT(type);
        ASSERT(size);
    }
    String m_name;
    GCGLenum m_type;
    GCGLint m_size;
};

} // namespace WebCore

#endif
