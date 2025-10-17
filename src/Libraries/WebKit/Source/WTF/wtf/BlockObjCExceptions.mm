/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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
#import "BlockObjCExceptions.h"

#import <wtf/Assertions.h>

void ReportBlockedObjCException(NSException *exception)
{
    // FIXME: This is probably going to be confusing when JavaScriptCore is used standalone. JSC
    // will call this code as part of default locale detection.
    // https://bugs.webkit.org/show_bug.cgi?id=157804
#if !ASSERT_ENABLED
    NSLog(@"*** WebKit discarding exception: <%@> %@", [exception name], [exception reason]);
#else
    ASSERT_WITH_MESSAGE(0, "Uncaught exception - %@", exception);
#endif
}
