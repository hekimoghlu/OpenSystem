/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
#import "ProcessCheck.h"

#import <Foundation/Foundation.h>
#import <mutex>

namespace bmalloc {

#if !BPLATFORM(WATCHOS)
bool gigacageEnabledForProcess()
{
    // Note that this function is only called once.
    // If we wanted to make it efficient to call more than once, we could memoize the result in a global boolean.

    NSString *appName = [[NSBundle mainBundle] bundleIdentifier];
    if (appName) {
        bool isWebProcess = [appName hasPrefix:@"com.apple.WebKit.WebContent"];
        return isWebProcess;
    }

    NSString *processName = [[NSProcessInfo processInfo] processName];
    bool isOptInBinary = [processName isEqualToString:@"jsc"]
        || [processName isEqualToString:@"DumpRenderTree"]
        || [processName isEqualToString:@"wasm"]
        || [processName hasPrefix:@"test"];

    return isOptInBinary;
}
#endif // !BPLATFORM(WATCHOS)

#if BUSE(CHECK_NANO_MALLOC)
bool shouldProcessUnconditionallyUseBmalloc()
{
    static bool result;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [&] () {
        if (NSString *appName = [[NSBundle mainBundle] bundleIdentifier]) {
            auto contains = [&] (NSString *string) {
                return [appName rangeOfString:string options:NSCaseInsensitiveSearch].location != NSNotFound;
            };
            result = contains(@"com.apple.WebKit") || contains(@"safari");
        } else {
            NSString *processName = [[NSProcessInfo processInfo] processName];
            result = [processName isEqualToString:@"jsc"] || [processName isEqualToString:@"wasm"];
        }
    });

    return result;
}
#endif // BUSE(CHECK_NANO_MALLOC)

}
