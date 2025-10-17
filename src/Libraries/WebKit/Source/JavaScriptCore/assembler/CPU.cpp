/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
#include "CPU.h"

#if (CPU(X86) || CPU(X86_64) || CPU(ARM64)) && OS(DARWIN)
#include <mutex>
#include <sys/sysctl.h>
#endif

#if ENABLE(ASSEMBLER)
#include "MacroAssembler.h"
#endif

namespace JSC {

#if PLATFORM(MAC) || PLATFORM(MACCATALYST)
bool isKernOpenSource()
{
    uint32_t val = 0;
    size_t valSize = sizeof(val);
    return !sysctlbyname("kern.opensource_kernel", &val, &valSize, nullptr, 0) && val;
}
#endif

#if (CPU(X86) || CPU(X86_64)) && OS(DARWIN)
bool isKernTCSMAvailable()
{
    if (!Options::useKernTCSM())
        return false;

    uint32_t val = 0;
    size_t valSize = sizeof(val);
    int rc = sysctlbyname("kern.tcsm_available", &val, &valSize, nullptr, 0);
    if (rc < 0)
        return false;
    return !!val;
}

bool enableKernTCSM()
{
    uint32_t val = 1;
    int rc = sysctlbyname("kern.tcsm_enable", nullptr, nullptr, &val, sizeof(val));
    if (rc < 0)
        return false;
    return true;
}

int kernTCSMAwareNumberOfProcessorCores()
{
    static std::once_flag onceFlag;
    static int result;
    std::call_once(onceFlag, [] {
        result = WTF::numberOfProcessorCores();
        if (result <= 1)
            return;
        if (isKernTCSMAvailable())
            --result;
    });
    return result;
}

int64_t hwL3CacheSize()
{
    int64_t val = 0;
    size_t valSize = sizeof(val);
    int rc = sysctlbyname("hw.l3cachesize", &val, &valSize, nullptr, 0);
    if (rc < 0)
        return 0;
    return val;
}

int32_t hwPhysicalCPUMax()
{
    int32_t val = 0;
    size_t valSize = sizeof(val);
    int rc = sysctlbyname("hw.physicalcpu_max", &val, &valSize, nullptr, 0);
    if (rc < 0)
        return 0;
    return val;
}

#endif // #if (CPU(X86) || CPU(X86_64)) && OS(DARWIN)

#if CPU(ARM64) && !(CPU(ARM64E) || OS(MAC_OS_X))
bool isARM64_LSE()
{
#if ENABLE(ASSEMBLER)
    return MacroAssembler::supportsLSE();
#else
    return false;
#endif
}
#endif

#if CPU(ARM64E)
bool isARM64E_FPAC()
{
#if OS(DARWIN)
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        uint32_t val = 0;
        size_t valSize = sizeof(val);
        int rc = sysctlbyname("hw.optional.arm.FEAT_FPAC", &val, &valSize, nullptr, 0);
        g_jscConfig.canUseFPAC = rc < 0 ? false : !!val;
    });
    return g_jscConfig.canUseFPAC;
#else
    return false;
#endif
}
#endif // CPU(ARM64E)

#if CPU(X86_64)
bool isX86_64_AVX()
{
#if ENABLE(ASSEMBLER)
    return MacroAssembler::supportsAVX();
#else
    return false;
#endif
}
#endif

} // namespace JSC
