/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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
#include <IOKit/system.h>
#include <IOKit/hidsystem/IOHIDShared.h>
#include "IOFixedPoint64.h"

//===========================================================================
/*  IOHIDSystemCursorHelper tracks 4 things.
        * The location of the cursor on the desktop. (location)
        * The change in location since the previous cursor update. (locationDelta)
        * The change in location that is currently being posted. (locationDeltaPosting)
        * The accumulated change since the last posting. (locationDeltaAccumulated)
    It also calculates one more.
        * The location of the cursor on the screen. (screenLocation)
 */
class IOHIDSystemCursorHelper
{
public:
    // the creator/destructor cannot be relied upon. use init/finalize instead.
    boolean_t       init();
    void            finalize()                      { }
    
    IOFixedPoint64& desktopLocation()               { return location; }
    IOFixedPoint64& desktopLocationDelta()          { return locationDelta; }
    IOFixedPoint64& desktopLocationPosting()        { return locationDeltaPosting; }
    void            applyPostingDelta();
    void            startPosting();
    IOFixedPoint64& desktopLocationAccumulated()    { return locationDeltaAccumulated; }
    void            incrementEventCount()           { eventCount++; }
    SInt32          getEventCount()                 { return eventCount; }
    SInt32          getEventCountPosting()          { return eventCountPosting; }
    void            clearEventCounts()              { eventCount = eventCountPosting = 0; }
    IOFixed64&      expectedCount()                 { return expectedCountValue; }
    
    IOFixedPoint64  getScreenLocation()             { return screenLocation; }
    void            updateScreenLocation(IOGBounds *desktop, 
                                         IOGBounds *screen);
    void            logPosition(const char *name, uint64_t ts);
    void            klogPosition(const char *name, uint64_t ts);
    
    bool            isPosting();

private:
    // all locations in desktop coordinates unless noted otherwise
    IOFixedPoint64  location;                   // aka pointerLoc
    IOFixedPoint64  locationDelta;              // aka pointerDelta
    IOFixedPoint64  locationDeltaPosting;       // aka postDeltaX/Y
    IOFixedPoint64  locationDeltaAccumulated;   // aka accumDX/Y
    
    IOFixedPoint64  screenLocation;
    
    IOFixed64       expectedCountValue;
    SInt32          eventCount;
    SInt32          eventCountPosting;
};

//===========================================================================
