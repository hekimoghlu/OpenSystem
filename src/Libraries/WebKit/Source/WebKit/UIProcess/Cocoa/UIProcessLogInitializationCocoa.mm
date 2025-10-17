/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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
#pragma once

#import "config.h"
#import "UIProcessLogInitialization.h"

#import <wtf/NeverDestroyed.h>
#import <wtf/text/WTFString.h>

#if PLATFORM(COCOA)

namespace WebKit {

namespace UIProcess {

String wtfLogLevelString()
{
    static NeverDestroyed<RetainPtr<NSString>> logString;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, [&] {
        logString.get() = [[NSUserDefaults standardUserDefaults] stringForKey:@"WTFLogging"];
    });
    return logString.get().get();
}

String webCoreLogLevelString()
{
    static NeverDestroyed<RetainPtr<NSString>> logString;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, [&] {
        logString.get() = [[NSUserDefaults standardUserDefaults] stringForKey:@"WebCoreLogging"];
    });
    return logString.get().get();
}

String webKitLogLevelString()
{
    static NeverDestroyed<RetainPtr<NSString>> logString;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, [&] {
        logString.get() = [[NSUserDefaults standardUserDefaults] stringForKey:@"WebKit2Logging"];
    });
    return logString.get().get();
}

}

}

#endif
