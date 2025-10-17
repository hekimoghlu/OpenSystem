/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 24, 2025.
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
#include "config.h"
#include "Logger.h"

#include <mutex>
#include <wtf/HexNumber.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/WTFString.h>

namespace WTF {

Lock loggerObserverLock;
Lock messageHandlerLoggerObserverLock;

String Logger::LogSiteIdentifier::toString() const
{
    if (className)
        return makeString(className, "::"_s, unsafeSpan(methodName), '(', hex(objectIdentifier), ") "_s);
    return makeString(unsafeSpan(methodName), '(', hex(objectIdentifier), ") "_s);
}

String LogArgument<const void*>::toString(const void* argument)
{
    return makeString('(', hex(reinterpret_cast<uintptr_t>(argument)), ')');
}

Vector<std::reference_wrapper<Logger::Observer>>& Logger::observers()
{
    static LazyNeverDestroyed<Vector<std::reference_wrapper<Observer>>> observers;
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        observers.construct();
    });
    return observers;
}

Vector<std::reference_wrapper<Logger::MessageHandlerObserver>>& Logger::messageHandlerObservers()
{
    static LazyNeverDestroyed<Vector<std::reference_wrapper<MessageHandlerObserver>>> observers;
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        observers.construct();
    });
    return observers;
}

} // namespace WTF
