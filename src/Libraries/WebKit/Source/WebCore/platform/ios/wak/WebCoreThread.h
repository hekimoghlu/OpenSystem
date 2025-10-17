/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
#ifndef WebCoreThread_h
#define WebCoreThread_h

#if TARGET_OS_IPHONE

#import <CoreGraphics/CoreGraphics.h>

// Use __has_include here so that things work when rewritten into WebKitLegacy headers.
#if __has_include(<WebCore/PlatformExportMacros.h>)
#import <WebCore/PlatformExportMacros.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif    
        
typedef struct {
    CGContextRef currentCGContext;
} WebThreadContext;
    
extern volatile bool webThreadShouldYield;
extern volatile unsigned webThreadDelegateMessageScopeCount;

#ifdef __OBJC__
@class NSRunLoop;
#else
class NSRunLoop;
#endif

// The lock is automatically freed at the bottom of the runloop. No need to unlock.
// Note that calling this function may hang your UI for several seconds. Don't use
// unless you have to.
WEBCORE_EXPORT void WebThreadLock(void);
    
// This is a no-op for compatibility only. It will go away. Please don't use.
WEBCORE_EXPORT void WebThreadUnlock(void);
    
// Please don't use anything below this line unless you know what you are doing. If unsure, ask.
// ---------------------------------------------------------------------------------------------
WEBCORE_EXPORT bool WebThreadIsLocked(void);
WEBCORE_EXPORT bool WebThreadIsLockedOrDisabled(void);
WEBCORE_EXPORT bool WebThreadIsLockedOrDisabledInMainOrWebThread(void);

WEBCORE_EXPORT void WebThreadLockPushModal(void);
WEBCORE_EXPORT void WebThreadLockPopModal(void);

WEBCORE_EXPORT void WebThreadEnable(void);
WEBCORE_EXPORT bool WebThreadIsEnabled(void);
WEBCORE_EXPORT bool WebThreadIsCurrent(void);
WEBCORE_EXPORT bool WebThreadNotCurrent(void);
    
// These are for <rdar://problem/6817341> Many apps crashing calling -[UIFieldEditor text] in secondary thread
// Don't use them to solve any random problems you might have.
WEBCORE_EXPORT void WebThreadLockFromAnyThread(void);
WEBCORE_EXPORT void WebThreadLockFromAnyThreadNoLog(void);
WEBCORE_EXPORT void WebThreadUnlockFromAnyThread(void);

// This is for <rdar://problem/8005192> Mail entered a state where message subject and content isn't displayed.
// It should only be used for MobileMail to work around <rdar://problem/8005192>.
WEBCORE_EXPORT void WebThreadUnlockGuardForMail(void);

static inline bool WebThreadShouldYield(void) { return webThreadShouldYield; }
static inline void WebThreadSetShouldYield(void) { webThreadShouldYield = true; }

WEBCORE_EXPORT NSRunLoop* WebThreadNSRunLoop(void);

#if defined(__cplusplus)
}
#endif

#endif // TARGET_OS_IPHONE

#endif // WebCoreThread_h
