/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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

#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace Profiler {

class ExecutionCounter {
    WTF_MAKE_TZONE_ALLOCATED(ExecutionCounter);
    WTF_MAKE_NONCOPYABLE(ExecutionCounter);
public:
    ExecutionCounter() : m_counter(0) { }
    
    uint64_t* address() { return &m_counter; }
    
    uint64_t count() const { return m_counter; }

private:
    uint64_t m_counter;
};

} } // namespace JSC::Profiler
