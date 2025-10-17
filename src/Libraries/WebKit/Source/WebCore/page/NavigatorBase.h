/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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

#include "ContextDestructionObserver.h"
#include "ExceptionOr.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class GPU;
class ScriptExecutionContext;
class ServiceWorkerContainer;
class StorageManager;
class WebCoreOpaqueRoot;
class WebLockManager;

class NavigatorBase : public RefCountedAndCanMakeWeakPtr<NavigatorBase>, public ContextDestructionObserver, public CanMakeCheckedPtr<NavigatorBase> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(NavigatorBase);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(NavigatorBase);
public:
    virtual ~NavigatorBase();

    static String appName();
    String appVersion() const;
    virtual const String& userAgent() const = 0;
    virtual String platform() const;

    static String appCodeName();
    static String product();
    static String productSub();
    static String vendor();
    static String vendorSub();

    virtual bool onLine() const = 0;

    static String language();
    static Vector<String> languages();

    StorageManager& storage();
    WebLockManager& locks();

    int hardwareConcurrency(ScriptExecutionContext&);

protected:
    explicit NavigatorBase(ScriptExecutionContext*);

private:
    RefPtr<StorageManager> m_storageManager;
    RefPtr<WebLockManager> m_webLockManager;

public:
    ServiceWorkerContainer& serviceWorker();
    ExceptionOr<ServiceWorkerContainer&> serviceWorker(ScriptExecutionContext&);

private:
    std::unique_ptr<ServiceWorkerContainer> m_serviceWorkerContainer;
};

WebCoreOpaqueRoot root(NavigatorBase*);

} // namespace WebCore
