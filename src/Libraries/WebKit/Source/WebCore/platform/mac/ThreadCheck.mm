/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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
#import "ThreadCheck.h"

#if PLATFORM(MAC)

#import <wtf/NeverDestroyed.h>
#import <wtf/RobinHoodHashSet.h>
#import <wtf/StdLibExtras.h>
#import <wtf/text/StringHash.h>

namespace WebCore {

static bool didReadThreadViolationBehaviorFromUserDefaults = false;
static bool threadViolationBehaviorIsDefault = true;
static std::array<ThreadViolationBehavior, MaximumThreadViolationRound> threadViolationBehavior { RaiseExceptionOnThreadViolation, RaiseExceptionOnThreadViolation, RaiseExceptionOnThreadViolation };

static void readThreadViolationBehaviorFromUserDefaults()
{
    didReadThreadViolationBehaviorFromUserDefaults = true;

    ThreadViolationBehavior newBehavior = LogOnFirstThreadViolation;
    NSString *threadCheckLevel = [[NSUserDefaults standardUserDefaults] stringForKey:@"WebCoreThreadCheck"];
    if (!threadCheckLevel)
        return;

    if ([threadCheckLevel isEqualToString:@"None"])
        newBehavior = NoThreadCheck;
    else if ([threadCheckLevel isEqualToString:@"Exception"])
        newBehavior = RaiseExceptionOnThreadViolation;
    else if ([threadCheckLevel isEqualToString:@"Log"])
        newBehavior = LogOnThreadViolation;
    else if ([threadCheckLevel isEqualToString:@"LogOnce"])
        newBehavior = LogOnFirstThreadViolation;
    else
        ASSERT_NOT_REACHED();

    threadViolationBehaviorIsDefault = false;

    for (unsigned i = 0; i < MaximumThreadViolationRound; ++i)
        threadViolationBehavior[i] = newBehavior;
}

void setDefaultThreadViolationBehavior(ThreadViolationBehavior behavior, ThreadViolationRound round)
{
    ASSERT(round < MaximumThreadViolationRound);
    if (round >= MaximumThreadViolationRound)
        return;
    if (!didReadThreadViolationBehaviorFromUserDefaults)
        readThreadViolationBehaviorFromUserDefaults();
    if (threadViolationBehaviorIsDefault)
        threadViolationBehavior[round] = behavior;
}

void reportThreadViolation(const char* function, ThreadViolationRound round)
{
    ASSERT(round < MaximumThreadViolationRound);
    if (round >= MaximumThreadViolationRound)
        return;
    if (!didReadThreadViolationBehaviorFromUserDefaults)
        readThreadViolationBehaviorFromUserDefaults();
    if (threadViolationBehavior[round] == NoThreadCheck)
        return;
    if (pthread_main_np())
        return;
    WebCoreReportThreadViolation(function, round);
}

} // namespace WebCore

// Split out the actual reporting of the thread violation to make it easier to set a breakpoint
void WebCoreReportThreadViolation(const char* function, WebCore::ThreadViolationRound round)
{
    using namespace WebCore;

    ASSERT(round < MaximumThreadViolationRound);
    if (round >= MaximumThreadViolationRound)
        return;

    static NeverDestroyed<MemoryCompactRobinHoodHashSet<String>> loggedFunctions;
    switch (threadViolationBehavior[round]) {
        case NoThreadCheck:
            break;
        case LogOnFirstThreadViolation:
            if (loggedFunctions.get().add(String::fromLatin1(function)).isNewEntry) {
                NSLog(@"WebKit Threading Violation - %s called from secondary thread", function);
                NSLog(@"Additional threading violations for this function will not be logged.");
            }
            break;
        case LogOnThreadViolation:
            NSLog(@"WebKit Threading Violation - %s called from secondary thread", function);
            break;
        case RaiseExceptionOnThreadViolation:
            [NSException raise:@"WebKitThreadingException" format:@"%s was called from a secondary thread", function];
            break;
    }
}

#endif // PLATFORM(MAC)
