/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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
#import "CompletionHandlerCallChecker.h"

#import <mutex>
#import <objc/runtime.h>
#import <wtf/Ref.h>
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>

namespace WebKit {

Ref<CompletionHandlerCallChecker> CompletionHandlerCallChecker::create(id delegate, SEL delegateMethodSelector)
{
    return adoptRef(*new CompletionHandlerCallChecker(object_getClass(delegate), delegateMethodSelector));
}

CompletionHandlerCallChecker::CompletionHandlerCallChecker(Class delegateClass, SEL delegateMethodSelector)
    : m_delegateClass(delegateClass)
    , m_delegateMethodSelector(delegateMethodSelector)
    , m_didCallCompletionHandler(false)
{
}

CompletionHandlerCallChecker::~CompletionHandlerCallChecker()
{
    if (m_didCallCompletionHandler)
        return;

    Class delegateClass = classImplementingDelegateMethod();
    [NSException raise:NSInternalInconsistencyException format:@"Completion handler passed to %c[%@ %@] was not called", class_isMetaClass(delegateClass) ? '+' : '-', NSStringFromClass(delegateClass), NSStringFromSelector(m_delegateMethodSelector)];
}

void CompletionHandlerCallChecker::didCallCompletionHandler()
{
    ASSERT(!m_didCallCompletionHandler);
    m_didCallCompletionHandler = true;
}

static bool shouldThrowExceptionForDuplicateCompletionHandlerCall()
{
    static bool shouldThrowException;
    static std::once_flag once;
    std::call_once(once, [] {
        shouldThrowException = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::ExceptionsForDuplicateCompletionHandlerCalls);
    });
    return shouldThrowException;
}

bool CompletionHandlerCallChecker::completionHandlerHasBeenCalled() const
{
    if (!m_didCallCompletionHandler)
        return false;

    if (shouldThrowExceptionForDuplicateCompletionHandlerCall()) {
        Class delegateClass = classImplementingDelegateMethod();
        [NSException raise:NSInternalInconsistencyException format:@"Completion handler passed to %c[%@ %@] was called more than once", class_isMetaClass(delegateClass) ? '+' : '-', NSStringFromClass(delegateClass), NSStringFromSelector(m_delegateMethodSelector)];
    }

    return true;
}

Class CompletionHandlerCallChecker::classImplementingDelegateMethod() const
{
    Class delegateClass = m_delegateClass;
    Method delegateMethod = class_getInstanceMethod(delegateClass, m_delegateMethodSelector);

    for (Class superclass = class_getSuperclass(delegateClass); superclass; superclass = class_getSuperclass(superclass)) {
        if (class_getInstanceMethod(superclass, m_delegateMethodSelector) != delegateMethod)
            break;

        delegateClass = superclass;
    }

    return delegateClass;
}

} // namespace WebKit
