/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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
#include <wtf/ApproximateTime.h>
#include <wtf/ContinuousApproximateTime.h>
#include <wtf/ContinuousTime.h>
#include <wtf/MonotonicTime.h>
#include <wtf/StdLibExtras.h>
#include <wtf/WallTime.h>

#if OS(DARWIN)
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mutex>
#include <sys/time.h>
#elif OS(WINDOWS)
#include <windows.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#else
#include <sys/time.h>
#include <time.h>
#endif

#if OS(FUCHSIA)
#include <zircon/syscalls.h>
#endif

#if OS(HAIKU)
#include <OS.h>
#endif

#if USE(GLIB)
#include <glib.h>
#endif

namespace WTF {

#if OS(WINDOWS)

// Number of 100 nanosecond between January 1, 1601 and January 1, 1970.
static constexpr ULONGLONG epochBias = 116444736000000000ULL;
static constexpr double hundredsOfNanosecondsPerMillisecond = 10000;

static double lowResUTCTime()
{
    FILETIME fileTime;

    GetSystemTimeAsFileTime(&fileTime);

    // As per Windows documentation for FILETIME, copy the resulting FILETIME structure to a
    // ULARGE_INTEGER structure using memcpy (using memcpy instead of direct assignment can
    // prevent alignment faults on 64-bit Windows).

    ULARGE_INTEGER dateTime;
    static_assert(sizeof(dateTime) == sizeof(fileTime));
    memcpySpan(asMutableByteSpan(dateTime), asByteSpan(fileTime));

    // Windows file times are in 100s of nanoseconds.
    return (dateTime.QuadPart - epochBias) / hundredsOfNanosecondsPerMillisecond;
}

static LARGE_INTEGER qpcFrequency;
static bool syncedTime;

static double highResUpTime()
{
    // We use QPC, but only after sanity checking its result, due to bugs:
    // http://support.microsoft.com/kb/274323
    // http://support.microsoft.com/kb/895980
    // http://msdn.microsoft.com/en-us/library/ms644904.aspx ("...you can get different results on different processors due to bugs in the basic input/output system (BIOS) or the hardware abstraction layer (HAL)."

    static LARGE_INTEGER qpcLast;
    static DWORD tickCountLast;
    static bool inited;

    LARGE_INTEGER qpc;
    QueryPerformanceCounter(&qpc);
#if defined(_M_IX86) || defined(__i386__)
    DWORD tickCount = GetTickCount();
#else
    ULONGLONG tickCount = GetTickCount64();
#endif

    if (inited) {
        __int64 qpcElapsed = ((qpc.QuadPart - qpcLast.QuadPart) * 1000) / qpcFrequency.QuadPart;
        __int64 tickCountElapsed;
        if (tickCount >= tickCountLast)
            tickCountElapsed = (tickCount - tickCountLast);
        else {
            __int64 tickCountLarge = tickCount + 0x100000000I64;
            tickCountElapsed = tickCountLarge - tickCountLast;
        }

        // force a re-sync if QueryPerformanceCounter differs from GetTickCount by more than 500ms.
        // (500ms value is from http://support.microsoft.com/kb/274323)
        __int64 diff = tickCountElapsed - qpcElapsed;
        if (diff > 500 || diff < -500)
            syncedTime = false;
    } else
        inited = true;

    qpcLast = qpc;
    tickCountLast = tickCount;

    return (1000.0 * qpc.QuadPart) / static_cast<double>(qpcFrequency.QuadPart);
}

static bool qpcAvailable()
{
    static bool available;
    static bool checked;

    if (checked)
        return available;

    available = QueryPerformanceFrequency(&qpcFrequency);
    checked = true;
    return available;
}

static inline double currentTime()
{
    // Use a combination of ftime and QueryPerformanceCounter.
    // ftime returns the information we want, but doesn't have sufficient resolution.
    // QueryPerformanceCounter has high resolution, but is only usable to measure time intervals.
    // To combine them, we call ftime and QueryPerformanceCounter initially. Later calls will use QueryPerformanceCounter
    // by itself, adding the delta to the saved ftime.  We periodically re-sync to correct for drift.
    static double syncLowResUTCTime;
    static double syncHighResUpTime;
    static double lastUTCTime;

    double lowResTime = lowResUTCTime();

    if (!qpcAvailable())
        return lowResTime / 1000.0;

    double highResTime = highResUpTime();

    if (!syncedTime) {
        timeBeginPeriod(1); // increase time resolution around low-res time getter
        syncLowResUTCTime = lowResTime = lowResUTCTime();
        timeEndPeriod(1); // restore time resolution
        syncHighResUpTime = highResTime;
        syncedTime = true;
    }

    double highResElapsed = highResTime - syncHighResUpTime;
    double utc = syncLowResUTCTime + highResElapsed;

    // force a clock re-sync if we've drifted
    double lowResElapsed = lowResTime - syncLowResUTCTime;
    const double maximumAllowedDriftMsec = 15.625 * 2.0; // 2x the typical low-res accuracy
    if (std::abs(highResElapsed - lowResElapsed) > maximumAllowedDriftMsec)
        syncedTime = false;

    // make sure time doesn't run backwards (only correct if difference is < 2 seconds, since DST or clock changes could occur)
    const double backwardTimeLimit = 2000.0;
    if (utc < lastUTCTime && (lastUTCTime - utc) < backwardTimeLimit)
        return lastUTCTime / 1000.0;
    lastUTCTime = utc;
    return utc / 1000.0;
}

Int128 currentTimeInNanoseconds()
{
    return static_cast<Int128>(currentTime() * 1'000'000'000);
}

#elif OS(HAIKU)

Int128 currentTimeInNanoseconds()
{
    return static_cast<Int128>(real_time_clock_usecs() * 1000.0);
}

double currentTime()
{
    return (double)real_time_clock_usecs() / 1'000'000.0;
}

#else

Int128 currentTimeInNanoseconds()
{
    struct timespec ts { };
    clock_gettime(CLOCK_REALTIME, &ts);
    return (static_cast<Int128>(ts.tv_sec) * 1'000'000'000) + ts.tv_nsec;
}

static inline double currentTime()
{
    struct timespec ts { };
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<double>(ts.tv_sec) + ts.tv_nsec / 1'000'000'000.0;
}

#endif

WallTime WallTime::now()
{
    return fromRawSeconds(currentTime());
}

#if OS(DARWIN)
static mach_timebase_info_data_t& machTimebaseInfo()
{
    // Based on listing #2 from Apple QA 1398, but modified to be thread-safe.
    static mach_timebase_info_data_t timebaseInfo;
    static std::once_flag initializeTimerOnceFlag;
    std::call_once(initializeTimerOnceFlag, [] {
        kern_return_t kr = mach_timebase_info(&timebaseInfo);
        ASSERT_UNUSED(kr, kr == KERN_SUCCESS);
        ASSERT(timebaseInfo.denom);
    });
    return timebaseInfo;
}

MonotonicTime MonotonicTime::fromMachAbsoluteTime(uint64_t machAbsoluteTime)
{
    auto& info = machTimebaseInfo();
    return fromRawSeconds((machAbsoluteTime * info.numer) / (1.0e9 * info.denom));
}

uint64_t MonotonicTime::toMachAbsoluteTime() const
{
    auto& info = machTimebaseInfo();
    return static_cast<uint64_t>((m_value * 1.0e9 * info.denom) / info.numer);
}

ApproximateTime ApproximateTime::fromMachApproximateTime(uint64_t machApproximateTime)
{
    auto& info = machTimebaseInfo();
    return fromRawSeconds((machApproximateTime * info.numer) / (1.0e9 * info.denom));
}

uint64_t ApproximateTime::toMachApproximateTime() const
{
    auto& info = machTimebaseInfo();
    return static_cast<uint64_t>((m_value * 1.0e9 * info.denom) / info.numer);
}

ContinuousTime ContinuousTime::fromMachContinuousTime(uint64_t machContinuousTime)
{
    auto& info = machTimebaseInfo();
    return fromRawSeconds((machContinuousTime * info.numer) / (1.0e9 * info.denom));
}

uint64_t ContinuousTime::toMachContinuousTime() const
{
    auto& info = machTimebaseInfo();
    return static_cast<uint64_t>((m_value * 1.0e9 * info.denom) / info.numer);
}

ContinuousApproximateTime ContinuousApproximateTime::fromMachContinuousApproximateTime(uint64_t machContinuousApproximateTime)
{
    auto& info = machTimebaseInfo();
    return fromRawSeconds((machContinuousApproximateTime * info.numer) / (1.0e9 * info.denom));
}

uint64_t ContinuousApproximateTime::toMachContinuousApproximateTime() const
{
    auto& info = machTimebaseInfo();
    return static_cast<uint64_t>((m_value * 1.0e9 * info.denom) / info.numer);
}
#endif

MonotonicTime MonotonicTime::now()
{
#if USE(GLIB)
    return fromRawSeconds(static_cast<double>(g_get_monotonic_time() / 1000000.0));
#elif OS(DARWIN)
    return fromMachAbsoluteTime(mach_absolute_time());
#elif OS(FUCHSIA)
    return fromRawSeconds(zx_clock_get_monotonic() / static_cast<double>(ZX_SEC(1)));
#elif OS(LINUX) || OS(FREEBSD) || OS(OPENBSD) || OS(NETBSD)
    struct timespec ts { };
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return fromRawSeconds(static_cast<double>(ts.tv_sec) + ts.tv_nsec / 1.0e9);
#elif OS(HAIKU)
    return fromRawSeconds(static_cast<double>(system_time_nsecs() / 1.0e9));
#else
    static double lastTime = 0;
    double currentTimeNow = currentTime();
    if (currentTimeNow < lastTime)
        return lastTime;
    lastTime = currentTimeNow;
    return fromRawSeconds(currentTimeNow);
#endif
}

ApproximateTime ApproximateTime::now()
{
#if OS(DARWIN)
    return fromMachApproximateTime(mach_approximate_time());
#elif OS(LINUX)
    struct timespec ts { };
    clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
    return fromRawSeconds(static_cast<double>(ts.tv_sec) + ts.tv_nsec / 1.0e9);
#elif OS(FREEBSD)
    struct timespec ts { };
    clock_gettime(CLOCK_MONOTONIC_FAST, &ts);
    return fromRawSeconds(static_cast<double>(ts.tv_sec) + ts.tv_nsec / 1.0e9);
#elif OS(HAIKU)
    return fromRawSeconds(static_cast<double>(system_time() / 1.0e6));
#else
    return ApproximateTime::fromRawSeconds(MonotonicTime::now().secondsSinceEpoch().value());
#endif
}

ContinuousTime ContinuousTime::now()
{
#if OS(DARWIN)
    return fromMachContinuousTime(mach_continuous_time());
#elif OS(LINUX) || OS(OPENBSD)
    struct timespec ts { };
    clock_gettime(CLOCK_BOOTTIME, &ts);
    return fromRawSeconds(static_cast<double>(ts.tv_sec) + ts.tv_nsec / 1.0e9);
#else
    static double lastTime = 0;
    double currentTimeNow = currentTime();
    if (currentTimeNow < lastTime)
        return lastTime;
    lastTime = currentTimeNow;
    return fromRawSeconds(currentTimeNow);
#endif
}

ContinuousApproximateTime ContinuousApproximateTime::now()
{
#if OS(DARWIN)
    return fromMachContinuousApproximateTime(mach_continuous_approximate_time());
#elif OS(LINUX) || OS(OPENBSD)
    struct timespec ts { };
    clock_gettime(CLOCK_BOOTTIME, &ts);
    return fromRawSeconds(static_cast<double>(ts.tv_sec) + ts.tv_nsec / 1.0e9);
#else
    static double lastTime = 0;
    double currentTimeNow = currentTime();
    if (currentTimeNow < lastTime)
        return lastTime;
    lastTime = currentTimeNow;
    return fromRawSeconds(currentTimeNow);
#endif
}

} // namespace WTF
