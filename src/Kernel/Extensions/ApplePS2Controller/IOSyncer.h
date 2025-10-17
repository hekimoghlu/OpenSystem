/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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
#ifndef _IOSYNCER_H
#define _IOSYNCER_H

#include <libkern/c++/OSObject.h>
#include <IOKit/IOTypes.h>
#include <IOKit/IOLocks.h>

class IOSyncer : public OSObject
{
    OSDeclareDefaultStructors(IOSyncer)

private:
    // The spin lock that is used to guard the 'threadMustStop' variable. 
    IOSimpleLock *guardLock;
    volatile bool threadMustStop;
    IOReturn fResult;
    virtual void free() override;
    virtual void privateSignal();

public:

    static IOSyncer * create(bool twoRetains = true);

    virtual bool init(bool twoRetains);
    virtual void reinit();
    virtual IOReturn wait(bool autoRelease = true);
    virtual void signal(IOReturn res = kIOReturnSuccess,
						bool autoRelease = true);
};

#endif /* !_IOSYNCER */

