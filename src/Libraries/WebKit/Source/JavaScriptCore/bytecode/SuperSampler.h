/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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

#include <atomic>

namespace JSC {

class MacroAssembler;

extern "C" JS_EXPORT_PRIVATE std::atomic<uint32_t> g_superSamplerCount;

void initializeSuperSampler();

class SuperSamplerScope {
public:
    SuperSamplerScope(bool doSample = true)
        : m_doSample(doSample)
    {
        if (m_doSample)
            ++g_superSamplerCount;
    }

    ~SuperSamplerScope()
    {
        if (m_doSample)
            --g_superSamplerCount;
    }

    void release()
    {
        ASSERT(m_doSample);
        --g_superSamplerCount;
        m_doSample = false;
    }

private:
    bool m_doSample;
};

JS_EXPORT_PRIVATE void resetSuperSamplerState();
JS_EXPORT_PRIVATE void printSuperSamplerState();
JS_EXPORT_PRIVATE void enableSuperSampler();
JS_EXPORT_PRIVATE void disableSuperSampler();

} // namespace JSC
