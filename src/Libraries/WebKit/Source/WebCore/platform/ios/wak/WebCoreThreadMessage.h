/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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
#ifndef WebCoreThreadMessage_h
#define WebCoreThreadMessage_h

#if TARGET_OS_IPHONE

#import <Foundation/Foundation.h>

#ifdef __OBJC__
#import <WebCore/WebCoreThread.h>
#endif // __OBJC__

#if defined(__cplusplus)
extern "C" {
#endif    

//
// Release an object on the main thread.
//
@interface NSObject(WebCoreThreadAdditions)
- (void)releaseOnMainThread;
@end

// Register a class for deallocation on the WebThread
WEBCORE_EXPORT void WebCoreObjCDeallocOnWebThread(Class cls);

// Asynchronous from main thread to web thread.
WEBCORE_EXPORT void WebThreadAdoptAndRelease(id obj);

// Synchronous from web thread to main thread, or main thread to main thread.
WEBCORE_EXPORT void WebThreadCallDelegate(NSInvocation *invocation);
WEBCORE_EXPORT void WebThreadRunOnMainThread(void (^)(void));

// Asynchronous from web thread to main thread, but synchronous when called on the main thread.
WEBCORE_EXPORT void WebThreadCallDelegateAsync(NSInvocation *invocation);

// Asynchronous from web thread to main thread, but synchronous when called on the main thread.
WEBCORE_EXPORT void WebThreadPostNotification(NSString *name, id object, id userInfo);

// Convenience method for making an NSInvocation object
WEBCORE_EXPORT NSInvocation *WebThreadMakeNSInvocation(id target, SEL selector);

#if defined(__cplusplus)
}
#endif

#endif // TARGET_OS_IPHONE

#endif // WebCoreThreadMessage_h
