/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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
#ifndef DenormalDisabler_h
#define DenormalDisabler_h

#include <cinttypes>
#include <wtf/MathExtras.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

// Deal with denormals. They can very seriously impact performance on x86.

// Define HAVE_DENORMAL if we support flushing denormals to zero.
#if OS(WINDOWS) && COMPILER(MSVC)
#define HAVE_DENORMAL
#endif

#if COMPILER(GCC_COMPATIBLE) && defined(__SSE__)
#define HAVE_DENORMAL
#endif

#ifdef HAVE_DENORMAL
class DenormalDisabler final {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(DenormalDisabler);
public:
    DenormalDisabler()
            : m_savedCSR(0)
    {
#if OS(WINDOWS) && COMPILER(MSVC)
        // Save the current state, and set mode to flush denormals.
        //
        // http://stackoverflow.com/questions/637175/possible-bug-in-controlfp-s-may-not-restore-control-word-correctly
        _controlfp_s(&m_savedCSR, 0, 0);
        unsigned int unused;
        _controlfp_s(&unused, _DN_FLUSH, _MCW_DN);
#else
        m_savedCSR = getCSR();
        setCSR(m_savedCSR | (isDAZSupported() ? 0x8040 : 0x8000));
#endif
    }

    ~DenormalDisabler()
    {
#if OS(WINDOWS) && COMPILER(MSVC)
        unsigned int unused;
        _controlfp_s(&unused, m_savedCSR, _MCW_DN);
#else
        setCSR(m_savedCSR);
#endif
    }

    // This is a nop if we can flush denormals to zero in hardware.
    static inline float flushDenormalFloatToZero(float f)
    {
#if OS(WINDOWS) && COMPILER(MSVC) && (!_M_IX86_FP)
        // For systems using x87 instead of sse, there's no hardware support
        // to flush denormals automatically. Hence, we need to flush
        // denormals to zero manually.
        return (std::abs(f) < FLT_MIN) ? 0.0f : f;
#else
        return f;
#endif
    }
private:
#if COMPILER(GCC_COMPATIBLE) && defined(__SSE__)
    static inline bool isDAZSupported()
    {
#if CPU(X86_64)
        return true;
#else
        static bool s_isInited = false;
        static bool s_isSupported = false;
        if (s_isInited)
            return s_isSupported;

        struct fxsaveResult {
            uint8_t before[28];
            uint32_t CSRMask;
            uint8_t after[480];
        } __attribute__ ((aligned (16)));

        fxsaveResult registerData;
        zeroBytes(registerData);
        asm volatile("fxsave %0" : "=m" (registerData));
        s_isSupported = registerData.CSRMask & 0x0040;
        s_isInited = true;
        return s_isSupported;
#endif
    }

    inline int getCSR()
    {
        int result;
        asm volatile("stmxcsr %0" : "=m" (result));
        return result;
    }

    inline void setCSR(int a)
    {
        int temp = a;
        asm volatile("ldmxcsr %0" : : "m" (temp));
    }

#endif

    unsigned int m_savedCSR;
};

#else
// FIXME: add implementations for other architectures and compilers
class DenormalDisabler final {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(DenormalDisabler);
public:
    DenormalDisabler() { }

    // Assume the worst case that other architectures and compilers
    // need to flush denormals to zero manually.
    static inline float flushDenormalFloatToZero(float f)
    {
        return (std::abs(f) < FLT_MIN) ? 0.0f : f;
    }
};

#endif

} // WebCore

#undef HAVE_DENORMAL
#endif // DenormalDisabler_h
