/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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

#include <glib.h>
#include <wtf/FastMalloc.h>

namespace JSC {

class JSCGLibWrapperObject {
    WTF_MAKE_FAST_ALLOCATED;
public:
    JSCGLibWrapperObject(gpointer object, GDestroyNotify destroyFunction)
        : m_object(object)
        , m_destroyFunction(destroyFunction)
    {
    }

    ~JSCGLibWrapperObject()
    {
        if (m_destroyFunction)
            m_destroyFunction(m_object);
    }

    gpointer object() const { return m_object; }

private:
    gpointer m_object { nullptr };
    GDestroyNotify m_destroyFunction { nullptr };
};

} // namespace JSC
