/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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
#import "config.h"
#import "SystemTracing.h"

#if HAVE(OS_SIGNPOST)

#import "ContinuousTime.h"
#import <dispatch/dispatch.h>
#import <mach/mach_time.h>

bool WTFSignpostIndirectLoggingEnabled;

os_log_t WTFSignpostLogHandle()
{
    static dispatch_once_t once;
    static os_log_t handle;

    dispatch_once(&once, ^{
        handle = os_log_create("com.apple.WebKit", "Signposts");
    });

    return handle;
}

#define WTF_SIGNPOST_EVENT_EMIT_FUNC_CASE_LABEL(emitFunc, name, timeFormat) \
    case WTFOSSignpostName ## name : \
        emitFunc(log, signpostIdentifier, #name, "pid: %d | %{public}s" timeFormat, pid, logString, timestamp); \
        break;

static void beginSignpostInterval(os_log_t log, WTFOSSignpostName signpostName, uint64_t signpostIdentifier, uint64_t timestamp, pid_t pid, const char* logString)
{
#define WTF_SIGNPOST_EVENT_BEGIN_INTERVAL_CASE_LABEL(name) \
    WTF_SIGNPOST_EVENT_EMIT_FUNC_CASE_LABEL(os_signpost_interval_begin, name, " %{signpost.description:begin_time}llu")

    switch (signpostName) {
        FOR_EACH_WTF_SIGNPOST_NAME(WTF_SIGNPOST_EVENT_BEGIN_INTERVAL_CASE_LABEL)
        default: break;
    }
}

static void endSignpostInterval(os_log_t log, WTFOSSignpostName signpostName, uint64_t signpostIdentifier, uint64_t timestamp, pid_t pid, const char* logString)
{
#define WTF_SIGNPOST_EVENT_END_INTERVAL_CASE_LABEL(name) \
    WTF_SIGNPOST_EVENT_EMIT_FUNC_CASE_LABEL(os_signpost_interval_end, name, " %{signpost.description:end_time}llu")

    switch (signpostName) {
        FOR_EACH_WTF_SIGNPOST_NAME(WTF_SIGNPOST_EVENT_END_INTERVAL_CASE_LABEL)
        default: break;
    }
}

static void emitSignpostEvent(os_log_t log, WTFOSSignpostName signpostName, uint64_t signpostIdentifier, uint64_t timestamp, pid_t pid, const char* logString)
{
#define WTF_SIGNPOST_EVENT_EMIT_CASE_LABEL(name) \
    WTF_SIGNPOST_EVENT_EMIT_FUNC_CASE_LABEL(os_signpost_event_emit, name, " %{signpost.description:event_time}llu")

    switch (signpostName) {
        FOR_EACH_WTF_SIGNPOST_NAME(WTF_SIGNPOST_EVENT_EMIT_CASE_LABEL)
        default: break;
    }
}

static void emitSignpost(os_log_t log, WTFOSSignpostType type, WTFOSSignpostName name, uint64_t signpostIdentifier, uint64_t timestamp, pid_t pid, const char* logString)
{
    switch (type) {
    case WTFOSSignpostTypeBeginInterval:
        beginSignpostInterval(log, name, signpostIdentifier, timestamp, pid, logString);
        break;
    case WTFOSSignpostTypeEndInterval:
        endSignpostInterval(log, name, signpostIdentifier, timestamp, pid, logString);
        break;
    case WTFOSSignpostTypeEmitEvent:
        emitSignpostEvent(log, name, signpostIdentifier, timestamp, pid, logString);
        break;
    default:
        break;
    }
}

bool WTFSignpostHandleIndirectLog(os_log_t log, pid_t pid, std::span<const char> nullTerminatedLogString)
{
    if (log != WTFSignpostLogHandle() || !nullTerminatedLogString.data())
        return false;

    int signpostType = 0;
    int signpostName = 0;
    uintptr_t signpostIdentifierPointer = 0;
    uint64_t timestamp = 0;
    int bytesConsumed = 0;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    if (sscanf(nullTerminatedLogString.data(), "type=%d name=%d p=%" SCNuPTR " ts=%llu %n", &signpostType, &signpostName, &signpostIdentifierPointer, &timestamp, &bytesConsumed) != 4)
        return false;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    if (signpostType < 0 || signpostType >= WTFOSSignpostTypeCount)
        return false;

    if (signpostName < 0 || signpostName >= WTFOSSignpostNameCount)
        return false;

    // Mix in bits from the pid, since pointers from different pids could be at the same address, causing signpost IDs to clash.
    signpostIdentifierPointer ^= pid;
    auto signpostIdentifier = os_signpost_id_make_with_pointer(log, reinterpret_cast<const void *>(signpostIdentifierPointer));

    emitSignpost(log, static_cast<WTFOSSignpostType>(signpostType), static_cast<WTFOSSignpostName>(signpostName), signpostIdentifier, timestamp, pid, nullTerminatedLogString.subspan(bytesConsumed).data());
    return true;
}

uint64_t WTFCurrentContinuousTime(Seconds deltaFromNow)
{
    return (ContinuousTime::now() + deltaFromNow).toMachContinuousTime();
}

#endif
