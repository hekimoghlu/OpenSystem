/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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

// This class implements "instance count" management for regression test coverage.
// Since it adds runtime overhead to manage the count variable, the actual
// functionality of the class is limited to debug builds.

namespace WTF {

template<typename T> class InstanceCounted {
public:
    static size_t instanceCount()
    {
#ifndef NDEBUG
        return s_instanceCount;
#else
        return 0;
#endif
    }

#ifndef NDEBUG
protected:
    InstanceCounted()
    {
        ++s_instanceCount;
    }

    InstanceCounted(const InstanceCounted&)
    {
        ++s_instanceCount;
    }

    InstanceCounted(InstanceCounted&&)
    {
        ++s_instanceCount;
    }

    ~InstanceCounted()
    {
        ASSERT(s_instanceCount);
        --s_instanceCount;
    }

private:
    static std::atomic_size_t s_instanceCount;
#endif // NDEBUG
};

#ifndef NDEBUG
template<typename T> std::atomic_size_t InstanceCounted<T>::s_instanceCount;
#endif

} // namespace WTF

using WTF::InstanceCounted;

