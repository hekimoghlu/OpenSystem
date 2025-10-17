/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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
#include "WebProcessExtensionManager.h"

#include "APIString.h"
#include "InjectedBundle.h"
#include "WebKitWebProcessExtensionPrivate.h"
#include <memory>
#include <wtf/FileSystem.h>
#include <wtf/text/CString.h>

namespace WebKit {

WebProcessExtensionManager& WebProcessExtensionManager::singleton()
{
    static NeverDestroyed<WebProcessExtensionManager> extensionManager;
    return extensionManager;
}

void WebProcessExtensionManager::scanModules(const String& webProcessExtensionsDirectory, Vector<String>& modules)
{
    auto moduleNames = FileSystem::listDirectory(webProcessExtensionsDirectory);
    for (auto& moduleName : moduleNames) {
        if (!moduleName.endsWith(".so"_s))
            continue;

        auto modulePath = FileSystem::pathByAppendingComponent(webProcessExtensionsDirectory, moduleName);
        if (FileSystem::fileExists(modulePath))
            modules.append(modulePath);
    }
}

static void parseUserData(API::Object* userData, String& webProcessExtensionsDirectory, GRefPtr<GVariant>& initializationUserData)
{
    ASSERT(userData->type() == API::Object::Type::String);

    CString userDataString = downcast<API::String>(userData)->string().utf8();

    WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GTK/WPE port
    GRefPtr<GVariant> variant = adoptGRef(g_variant_parse(nullptr, userDataString.data(),
        userDataString.data() + userDataString.length(), nullptr, nullptr));
    WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    ASSERT(variant);
    ASSERT(g_variant_check_format_string(variant.get(), "(m&smv)", FALSE));

    const char* directory = nullptr;
    GVariant* data = nullptr;
    g_variant_get(variant.get(), "(m&smv)", &directory, &data);

    webProcessExtensionsDirectory = FileSystem::stringFromFileSystemRepresentation(directory);
    initializationUserData = adoptGRef(data);
}

bool WebProcessExtensionManager::initializeWebProcessExtension(Module* extensionModule, GVariant* userData)
{
#if ENABLE(2022_GLIB_API)
    WebKitWebProcessExtensionInitializeWithUserDataFunction initializeWithUserDataFunction =
        extensionModule->functionPointer<WebKitWebProcessExtensionInitializeWithUserDataFunction>("webkit_web_process_extension_initialize_with_user_data");
#else
    WebKitWebExtensionInitializeWithUserDataFunction initializeWithUserDataFunction =
        extensionModule->functionPointer<WebKitWebExtensionInitializeWithUserDataFunction>("webkit_web_extension_initialize_with_user_data");
#endif
    if (initializeWithUserDataFunction) {
        initializeWithUserDataFunction(m_extension.get(), userData);
        return true;
    }

#if ENABLE(2022_GLIB_API)
    WebKitWebProcessExtensionInitializeFunction initializeFunction =
        extensionModule->functionPointer<WebKitWebProcessExtensionInitializeFunction>("webkit_web_process_extension_initialize");
#else
    WebKitWebExtensionInitializeFunction initializeFunction =
        extensionModule->functionPointer<WebKitWebExtensionInitializeFunction>("webkit_web_extension_initialize");
#endif
    if (initializeFunction) {
        initializeFunction(m_extension.get());
        return true;
    }

    return false;
}

void WebProcessExtensionManager::initialize(InjectedBundle* bundle, API::Object* userDataObject)
{
    ASSERT(bundle);
    ASSERT(userDataObject);
    m_extension = adoptGRef(webkitWebProcessExtensionCreate(bundle));

    String webProcessExtensionsDirectory;
    GRefPtr<GVariant> userData;
    parseUserData(userDataObject, webProcessExtensionsDirectory, userData);

    if (webProcessExtensionsDirectory.isNull())
        return;

    Vector<String> modulePaths;
    scanModules(webProcessExtensionsDirectory, modulePaths);

    for (size_t i = 0; i < modulePaths.size(); ++i) {
        auto module = makeUnique<Module>(modulePaths[i]);
        if (!module->load())
            continue;
        if (initializeWebProcessExtension(module.get(), userData.get()))
            m_extensionModules.append(module.release());
    }
}

} // namespace WebKit
