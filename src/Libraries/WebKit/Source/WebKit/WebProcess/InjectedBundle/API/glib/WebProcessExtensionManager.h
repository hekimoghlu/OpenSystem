/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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

#include "Module.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/Noncopyable.h>
#include <wtf/Vector.h>
#include <wtf/glib/GRefPtr.h>

#if ENABLE(2022_GLIB_API)
#include "WebKitWebProcessExtension.h"
#else
#include "WebKitWebExtension.h"
typedef _WebKitWebExtension WebKitWebProcessExtension;
#endif

namespace API {
class Object;
}

namespace WebKit {

class InjectedBundle;

class WebProcessExtensionManager {
    WTF_MAKE_NONCOPYABLE(WebProcessExtensionManager);
public:
    __attribute__((visibility("default"))) static WebProcessExtensionManager& singleton();

    __attribute__((visibility("default"))) void initialize(InjectedBundle*, API::Object*);

    WebKitWebProcessExtension* extension() const { return m_extension.get(); }

private:
    WebProcessExtensionManager() = default;

    void scanModules(const String&, Vector<String>&);
    bool initializeWebProcessExtension(Module* extensionModule, GVariant* userData);

    Vector<Module*> m_extensionModules;
    GRefPtr<WebKitWebProcessExtension> m_extension;

    friend NeverDestroyed<WebProcessExtensionManager>;
};

} // namespace WebKit
