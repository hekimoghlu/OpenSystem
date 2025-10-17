/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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

#include "ScreenOrientationLockType.h"
#include "ScreenOrientationType.h"
#include <wtf/CompletionHandler.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class ScreenOrientationManager;
class ScreenOrientationManagerObserver;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::ScreenOrientationManager> : std::true_type { };
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::ScreenOrientationManagerObserver> : std::true_type { };
}

namespace WebCore {

class DeferredPromise;
class Exception;
class ScreenOrientation;

class ScreenOrientationManagerObserver : public CanMakeWeakPtr<ScreenOrientationManagerObserver> {
public:
    virtual ~ScreenOrientationManagerObserver() { }
    virtual void screenOrientationDidChange(ScreenOrientationType) = 0;
};

class ScreenOrientationManager : public CanMakeWeakPtr<ScreenOrientationManager> {
public:
    WEBCORE_EXPORT virtual ~ScreenOrientationManager();

    virtual ScreenOrientationType currentOrientation() = 0;
    virtual void lock(ScreenOrientationLockType, CompletionHandler<void(std::optional<Exception>&&)>&&) = 0;
    virtual void unlock() = 0;
    virtual void addObserver(ScreenOrientationManagerObserver&) = 0;
    virtual void removeObserver(ScreenOrientationManagerObserver&) = 0;

    void setLockPromise(ScreenOrientation&, Ref<DeferredPromise>&&);
    ScreenOrientation* lockRequester() const;
    RefPtr<DeferredPromise> takeLockPromise();

protected:
    WEBCORE_EXPORT ScreenOrientationManager();

private:
    RefPtr<DeferredPromise> m_lockPromise;
    WeakPtr<ScreenOrientation> m_lockRequester;
};

} // namespace WebCore
