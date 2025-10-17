/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 14, 2022.
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

#include "Node.h"

namespace WebCore {

class WebCoreOpaqueRoot {
public:
    template<typename T, typename = typename std::enable_if_t<!std::is_same_v<T, void>>>
    explicit WebCoreOpaqueRoot(T* pointer)
        : m_pointer(static_cast<void*>(pointer))
    {
    }

    WebCoreOpaqueRoot(std::nullptr_t) { }

    void* pointer() const { return m_pointer; }

private:
    void* m_pointer { nullptr };
};

template<typename Visitor>
inline void addWebCoreOpaqueRoot(Visitor&, WebCoreOpaqueRoot);

template<typename Visitor, typename ImplType>
inline void addWebCoreOpaqueRoot(Visitor&, ImplType*);

template<typename Visitor, typename ImplType>
inline void addWebCoreOpaqueRoot(Visitor&, ImplType&);

template<typename Visitor>
inline bool containsWebCoreOpaqueRoot(Visitor&, WebCoreOpaqueRoot);

template<typename Visitor, typename ImplType>
inline bool containsWebCoreOpaqueRoot(Visitor&, ImplType&);

template<typename Visitor, typename ImplType>
inline bool containsWebCoreOpaqueRoot(Visitor&, ImplType*);

} // namespace WebCore
