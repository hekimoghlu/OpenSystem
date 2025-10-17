/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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
#import "KillRing.h"
#import <wtf/TZoneMallocInlines.h>

#if PLATFORM(MAC)

namespace PAL {

WTF_MAKE_TZONE_ALLOCATED_IMPL(KillRing);

extern "C" {

// Kill ring calls. Would be better to use NSKillRing.h, but that's not available as API or SPI.

void _NSInitializeKillRing();
void _NSAppendToKillRing(NSString *);
void _NSPrependToKillRing(NSString *);
NSString *_NSYankFromKillRing();
void _NSNewKillRingSequence();
void _NSSetKillRingToYankedState();
void _NSResetKillRingOperationFlag();

}

static void initializeKillRingIfNeeded()
{
    static bool initializedKillRing = false;
    if (!initializedKillRing) {
        initializedKillRing = true;
        _NSInitializeKillRing();
    }
}

void KillRing::append(const String& string)
{
    initializeKillRingIfNeeded();
    // Necessary to prevent an implicit new sequence if the previous command was NSPrependToKillRing.
    _NSResetKillRingOperationFlag();
    _NSAppendToKillRing(string);
}

void KillRing::prepend(const String& string)
{
    initializeKillRingIfNeeded();
    // Necessary to prevent an implicit new sequence if the previous command was NSAppendToKillRing.
    _NSResetKillRingOperationFlag();
    _NSPrependToKillRing(string);
}

String KillRing::yank()
{
    initializeKillRingIfNeeded();
    return _NSYankFromKillRing();
}

void KillRing::startNewSequence()
{
    initializeKillRingIfNeeded();
    _NSNewKillRingSequence();
}

void KillRing::setToYankedState()
{
    initializeKillRingIfNeeded();
    _NSSetKillRingToYankedState();
}

} // namespace PAL

#endif // PLATFORM(MAC)
