/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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

#include "IDLTypes.h"
#include <JavaScriptCore/Strong.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace JSC {
class JSObject;
} // namespace JSC

namespace WTF {
class String;
}

namespace WebCore {

class NavigatorBase;
class PermissionStatus;
class ScriptExecutionContext;
enum class PermissionName : uint8_t;
enum class PermissionQuerySource : uint8_t;

template<typename IDLType> class DOMPromiseDeferred;

class Permissions : public RefCounted<Permissions> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Permissions);
public:
    static Ref<Permissions> create(NavigatorBase&);
    ~Permissions();

    NavigatorBase* navigator();
    void query(JSC::Strong<JSC::JSObject>, DOMPromiseDeferred<IDLInterface<PermissionStatus>>&&);
    WEBCORE_EXPORT static std::optional<PermissionQuerySource> sourceFromContext(const ScriptExecutionContext&);
    WEBCORE_EXPORT static std::optional<PermissionName> toPermissionName(const String&);

private:
    explicit Permissions(NavigatorBase&);

    WeakPtr<NavigatorBase> m_navigator;
};

} // namespace WebCore
