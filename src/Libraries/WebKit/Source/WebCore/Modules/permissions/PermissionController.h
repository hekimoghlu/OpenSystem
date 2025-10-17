/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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

#include "PermissionDescriptor.h"
#include <wtf/CompletionHandler.h>
#include <wtf/RefPtr.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

enum class PermissionName : uint8_t;
enum class PermissionQuerySource : uint8_t;
enum class PermissionState : uint8_t;
class Page;
class PermissionObserver;
struct ClientOrigin;
class SecurityOriginData;

class PermissionController : public ThreadSafeRefCounted<PermissionController> {
public:
    static PermissionController& shared();
    WEBCORE_EXPORT static void setSharedController(Ref<PermissionController>&&);
    
    virtual ~PermissionController() = default;
    virtual void query(ClientOrigin&&, PermissionDescriptor, const WeakPtr<Page>&, PermissionQuerySource, CompletionHandler<void(std::optional<PermissionState>)>&&) = 0;
    virtual void addObserver(PermissionObserver&) = 0;
    virtual void removeObserver(PermissionObserver&) = 0;
    virtual void permissionChanged(PermissionName, const SecurityOriginData&) = 0;
protected:
    PermissionController() = default;
};

class DummyPermissionController final : public PermissionController {
public:
    static Ref<DummyPermissionController> create() { return adoptRef(*new DummyPermissionController); }
private:
    DummyPermissionController() = default;
    void query(ClientOrigin&&, PermissionDescriptor, const WeakPtr<Page>&, PermissionQuerySource, CompletionHandler<void(std::optional<PermissionState>)>&& callback) final { callback({ }); }
    void addObserver(PermissionObserver&) final { }
    void removeObserver(PermissionObserver&) final { }
    void permissionChanged(PermissionName, const SecurityOriginData&) final { }
};

} // namespace WebCore
