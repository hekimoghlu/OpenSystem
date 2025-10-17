/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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

#if ENABLE(WK_WEB_EXTENSIONS)

#include "JSWebExtensionWrappable.h"
#include "JSWebExtensionWrapper.h"
#include "WebExtensionContentWorldType.h"
#include "WebExtensionContextProxy.h"
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/text/MakeString.h>

namespace WebKit {

class WebExtensionAPIRuntime;
class WebExtensionAPIWebPageRuntime;

class WebExtensionAPIObject {
public:
    WebExtensionAPIObject(WebExtensionContentWorldType contentWorldType)
        : m_contentWorldType(contentWorldType)
    {
        // This should only be called when creating a namespace object for web pages.
        ASSERT(contentWorldType == WebExtensionContentWorldType::WebPage);
    }

    WebExtensionAPIObject(WebExtensionContentWorldType contentWorldType, WebExtensionContextProxy& context)
        : m_contentWorldType(contentWorldType)
        , m_extensionContext(&context)
    {
    }

    WebExtensionAPIObject(WebExtensionContentWorldType contentWorldType, WebExtensionAPIRuntimeBase& runtime, WebExtensionContextProxy& context)
        : m_contentWorldType(contentWorldType)
        , m_runtime(&runtime)
        , m_extensionContext(&context)
    {
    }

    WebExtensionAPIObject(const WebExtensionAPIObject& parentObject)
        : m_contentWorldType(parentObject.contentWorldType())
        , m_runtime(&parentObject.runtime())
        , m_extensionContext(parentObject.m_extensionContext) // Using parentObject.extensionContext() is not safe for APIWebPage objects.
    {
    }

    virtual ~WebExtensionAPIObject() = default;

    bool isForMainWorld() const { return m_contentWorldType == WebExtensionContentWorldType::Main; }
    WebExtensionContentWorldType contentWorldType() const { return m_contentWorldType; }

    virtual WebExtensionAPIRuntimeBase& runtime() const { return *m_runtime; }

    WebExtensionContextProxy& extensionContext() const { return *m_extensionContext; }
    bool hasExtensionContext() const { return !!m_extensionContext; }

    const String& propertyPath() const { return m_propertyPath; }
    void setPropertyPath(const String& propertyName, const WebExtensionAPIObject* parentObject = nullptr)
    {
        ASSERT(!propertyName.isEmpty());

        if (parentObject && !parentObject->propertyPath().isEmpty())
            m_propertyPath = makeString(parentObject->propertyPath(), '.', propertyName);
        else
            m_propertyPath = propertyName;
    }

private:
    WebExtensionContentWorldType m_contentWorldType { WebExtensionContentWorldType::Main };
    RefPtr<WebExtensionAPIRuntimeBase> m_runtime;
    RefPtr<WebExtensionContextProxy> m_extensionContext;
    String m_propertyPath;
};

} // namespace WebKit

#define WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(ImplClass, ScriptClass, PropertyName) \
public: \
    template<typename... Args> \
    static Ref<ImplClass> create(Args&&... args) \
    { \
        return adoptRef(*new ImplClass(std::forward<Args>(args)...)); \
    } \
\
private: \
    explicit ImplClass(WebExtensionContentWorldType contentWorldType) \
        : WebExtensionAPIObject(contentWorldType) \
    { \
        setPropertyPath(#PropertyName##_s); \
    } \
\
    explicit ImplClass(WebExtensionContentWorldType contentWorldType, WebExtensionContextProxy& context) \
        : WebExtensionAPIObject(contentWorldType, context) \
    { \
        setPropertyPath(#PropertyName##_s); \
    } \
\
    explicit ImplClass(WebExtensionContentWorldType contentWorldType, WebExtensionAPIRuntimeBase& runtime, WebExtensionContextProxy& context) \
        : WebExtensionAPIObject(contentWorldType, runtime, context) \
    { \
        setPropertyPath(#PropertyName##_s); \
    } \
\
    explicit ImplClass(const WebExtensionAPIObject& parentObject) \
        : WebExtensionAPIObject(parentObject) \
    { \
        setPropertyPath(#PropertyName##_s, &parentObject); \
    } \
\
    JSClassRef wrapperClass() final { return JS##ImplClass::ScriptClass##Class(); } \
\
    using __thisIsHereToForceASemicolonAfterThisMacro UNUSED_TYPE_ALIAS = int

// End of macro.

// Needs to be at the end to allow WebExtensionAPIRuntime.h to include this header
// and avoid having all APIs need to include it.
#include "WebExtensionAPIRuntime.h"

#endif // ENABLE(WK_WEB_EXTENSIONS)
