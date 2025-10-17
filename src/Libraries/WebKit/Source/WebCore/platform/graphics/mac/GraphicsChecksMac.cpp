/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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
#include "GraphicsChecksMac.h"

#if PLATFORM(MAC)

#if HAVE(APPLE_GRAPHICS_CONTROL)
#include <sys/sysctl.h>
#endif

namespace WebCore {

#if HAVE(APPLE_GRAPHICS_CONTROL)

enum {
    kAGCOpen,
    kAGCClose
};

static io_connect_t attachToAppleGraphicsControl()
{
    mach_port_t mainPort;

    if (IOMainPort(MACH_PORT_NULL, &mainPort) != KERN_SUCCESS)
        return IO_OBJECT_NULL;

    CFDictionaryRef classToMatch = IOServiceMatching("AppleGraphicsControl");
    if (!classToMatch)
        return IO_OBJECT_NULL;

    kern_return_t kernResult;
    io_iterator_t iterator;
    if ((kernResult = IOServiceGetMatchingServices(mainPort, classToMatch, &iterator)) != KERN_SUCCESS)
        return IO_OBJECT_NULL;

    io_service_t serviceObject = IOIteratorNext(iterator);
    IOObjectRelease(iterator);
    if (!serviceObject)
        return IO_OBJECT_NULL;

    io_connect_t dataPort;
    kernResult = IOServiceOpen(serviceObject, mach_task_self(), 0, &dataPort);

    return (kernResult == KERN_SUCCESS) ? dataPort : IO_OBJECT_NULL;
}

static bool hasMuxCapability()
{
    io_connect_t dataPort = attachToAppleGraphicsControl();

    if (dataPort == IO_OBJECT_NULL)
        return false;

    bool result;
    if (IOConnectCallScalarMethod(dataPort, kAGCOpen, nullptr, 0, nullptr, nullptr) == KERN_SUCCESS) {
        IOConnectCallScalarMethod(dataPort, kAGCClose, nullptr, 0, nullptr, nullptr);
        result = true;
    } else
        result = false;

    IOServiceClose(dataPort);

    if (result) {
        // This is detecting Mac hardware with an Intel g575 GPU, which
        // we don't want to make available to muxing.
        // Based on information from Apple's OpenGL team, such devices
        // have four or fewer processors.
        // <rdar://problem/30060378>
        int names[2] = { CTL_HW, HW_NCPU };
        int cpuCount;
        size_t cpuCountLength = sizeof(cpuCount);
        sysctl(names, 2, &cpuCount, &cpuCountLength, nullptr, 0);
        result = cpuCount > 4;
    }

    return result;
}

#endif // HAVE(APPLE_GRAPHICS_CONTROL)

bool hasLowAndHighPowerGPUs()
{
#if HAVE(APPLE_GRAPHICS_CONTROL)
    static bool canMux = hasMuxCapability();
    return canMux;
#else
    return false;
#endif
}

}

#endif

