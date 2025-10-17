/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
#include <IOKit/IOLib.h>
#include "IOHIDSystemCursorHelper.h"
#include "IOHIDDebug.h"

//===========================================================================
boolean_t IOHIDSystemCursorHelper::init()
{
    location.fromIntFloor(0, 0);
    locationDelta.fromIntFloor(0, 0);
    locationDeltaPosting.fromIntFloor(0, 0);
    locationDeltaAccumulated.fromIntFloor(0, 0);
    screenLocation.fromIntFloor(0, 0);
    expectedCountValue.fromIntFloor(0);
    eventCount = 0;
    eventCountPosting = 0;

    return true;
}

//===========================================================================
void IOHIDSystemCursorHelper::startPosting()
{
    locationDeltaPosting = locationDeltaAccumulated;
    locationDeltaAccumulated.fromIntFloor(0, 0);
    eventCountPosting = eventCount;
    eventCount = 0;
}

//===========================================================================
void IOHIDSystemCursorHelper::updateScreenLocation(IOGBounds *desktop,
                                                   IOGBounds *screen)
{
    if (!desktop || !screen) {
        // no transform can be performed
        screenLocation = location;
    }
    else {
        if (*(UInt64*)desktop == *(UInt64*)screen) {
            //  no transform needed
            screenLocation = location;
        }
        else {
            int screenWidth   = screen->maxx  - screen->minx;
            int screenHeight  = screen->maxy  - screen->miny;
            int desktopWidth  = desktop->maxx - desktop->minx;
            int desktopHeight = desktop->maxy - desktop->miny;
            IOFixedPoint64 scratch;
            if ((screenWidth <= 0) || (screenHeight <= 0) || (desktopWidth <= 0) || (desktopHeight <= 0)) {
                // no transform can be performed
            }
            else {
                if ((screenWidth == desktopWidth) && (screenHeight == desktopHeight)) {
                    // translation only
                    screenLocation = location;
                    screenLocation += scratch.fromIntFloor(screen->minx - desktop->minx,
                                                           screen->miny - desktop->miny);
                }
                else {
                    // full transform
                    IOFixed64 x_scale;
                    IOFixed64 y_scale;
                    x_scale.fromIntFloor(screenWidth) /= desktopWidth;
                    y_scale.fromIntFloor(screenHeight) /= desktopHeight;
                    screenLocation = location;
                    screenLocation -= scratch.fromIntFloor(desktop->minx, desktop->miny);
                    screenLocation *= scratch.fromFixed64(x_scale, y_scale);
                    screenLocation += scratch.fromIntFloor(screen->minx, screen->miny);
                }
            }
        }
    }
}

//===========================================================================
void IOHIDSystemCursorHelper::applyPostingDelta()
{
    if (eventCountPosting && expectedCountValue) {
        // unless the eventCountPosting is within the expectedCount Â± 1, this adjustment
        // will cause more harm than good.
        if ((expectedCountValue <= eventCountPosting + 1LL) &&
            (expectedCountValue >= eventCountPosting - 1LL)) {
            locationDeltaPosting *= expectedCountValue;
            locationDeltaPosting /= eventCountPosting;
        }
    }
    eventCountPosting = 0;
    location += locationDeltaPosting;
    locationDelta += locationDeltaPosting;
    locationDeltaPosting.fromIntFloor(0, 0);
}


//===========================================================================
bool IOHIDSystemCursorHelper::isPosting()
{
    bool result = false;
    if (locationDeltaPosting) {
        result = true;
    }
    return result;
}

//===========================================================================
void IOHIDSystemCursorHelper::logPosition(const char *name, uint64_t ts)
{
    HIDLog("IOHIDSystem::%-20s cursor @ %lld: "
          "(%4lld.%02lld, %4lld.%02lld) "
          "[%+3lld.%02lld, %+3lld.%02lld] "
          "P[%+3lld.%02lld, %+3lld.%02lld] "
          "A[%+3lld.%02lld, %+3lld.%02lld] "
          "S(%4lld.%02lld, %4lld.%02lld) "
          "C %ld/%ld of %lld.%01lld",
          name, ts,
          location.xValue().as64(), (location.xValue().asFixed64() & 0xffff) / 656,
          location.yValue().as64(), (location.yValue().asFixed64() & 0xffff) / 656,
          locationDelta.xValue().as64(), (locationDelta.xValue().asFixed64() & 0xffff) / 656,
          locationDelta.yValue().as64(), (locationDelta.yValue().asFixed64() & 0xffff) / 656,
          locationDeltaPosting.xValue().as64(), (locationDeltaPosting.xValue().asFixed64() & 0xffff) / 656,
          locationDeltaPosting.yValue().as64(), (locationDeltaPosting.yValue().asFixed64() & 0xffff) / 656,
          locationDeltaAccumulated.xValue().as64(), (locationDeltaAccumulated.xValue().asFixed64() & 0xffff) / 656,
          locationDeltaAccumulated.yValue().as64(), (locationDeltaAccumulated.yValue().asFixed64() & 0xffff) / 656,
          screenLocation.xValue().as64(), (screenLocation.xValue().asFixed64() & 0xffff) / 656,
          screenLocation.yValue().as64(), (screenLocation.yValue().asFixed64() & 0xffff) / 656,
          (long int)eventCount,
          (long int)eventCountPosting,
          expectedCountValue.as64(), (expectedCountValue.asFixed64() & 0xffff) / 6554
          );
}

//===========================================================================
void IOHIDSystemCursorHelper::klogPosition(const char *name, uint64_t ts)
{
    kprintf("IOHIDSystem::%-20s cursor @ %lld: "
            "(%016llx, %016llx) "
            "[%016llx, %016llx] "
            "P[%016llx, %016llx] "
            "A[%016llx, %016llx] "
            "S(%016llx, %016llx) "
            "C %ld/%ld of %016llx\n",
            name, ts,
            location.xValue().asFixed64(),
            location.yValue().asFixed64(),
            locationDelta.xValue().asFixed64(),
            locationDelta.yValue().asFixed64(),
            locationDeltaPosting.xValue().asFixed64(),
            locationDeltaPosting.yValue().asFixed64(),
            locationDeltaAccumulated.xValue().asFixed64(),
            locationDeltaAccumulated.yValue().asFixed64(),
            screenLocation.xValue().asFixed64(),
            screenLocation.yValue().asFixed64(),
            (long int)eventCount,
            (long int)eventCountPosting,
            expectedCountValue.asFixed64()
            );
}

//===========================================================================
