/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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
#import "ScriptController.h"

#import "BridgeJSC.h"
#import "CommonVM.h"
#import "FrameLoader.h"
#import "LocalDOMWindow.h"
#import "LocalFrame.h"
#import "LocalFrameLoaderClient.h"
#import "WebScriptObjectPrivate.h"
#import "Widget.h"
#import "objc_instance.h"
#import "runtime_root.h"
#import <JavaScriptCore/APICast.h>
#import <JavaScriptCore/JSContextInternal.h>
#import <JavaScriptCore/JSLock.h>

@interface NSObject (WebPlugin)
- (id)objectForWebScript;
- (RefPtr<JSC::Bindings::Instance>)createPluginBindingsInstance:(Ref<JSC::Bindings::RootObject>&&)rootObject;
@end

using namespace JSC::Bindings;

namespace WebCore {

RefPtr<JSC::Bindings::Instance> ScriptController::createScriptInstanceForWidget(Widget* widget)
{
    NSView* widgetView = widget->platformWidget();
    if (!widgetView)
        return nullptr;

    auto rootObject = createRootObject((__bridge void*)widgetView);

    if ([widgetView respondsToSelector:@selector(createPluginBindingsInstance:)])
        return [widgetView createPluginBindingsInstance:WTFMove(rootObject)];
        
    if ([widgetView respondsToSelector:@selector(objectForWebScript)]) {
        id objectForWebScript = [widgetView objectForWebScript];
        if (!objectForWebScript)
            return nullptr;
        return JSC::Bindings::ObjcInstance::create(objectForWebScript, WTFMove(rootObject));
    }
    return nullptr;
}

WebScriptObject *ScriptController::windowScriptObject()
{
    if (!canExecuteScripts(ReasonForCallingCanExecuteScripts::NotAboutToExecuteScript))
        return nil;

    if (!m_windowScriptObject) {
        JSC::JSLockHolder lock(commonVM());
        JSC::Bindings::RootObject* root = bindingRootObject();
        m_windowScriptObject = [WebScriptObject scriptObjectForJSObject:toRef(&jsWindowProxy(pluginWorld())) originRootObject:root rootObject:root];
    }

    return m_windowScriptObject.get();
}

JSContext *ScriptController::javaScriptContext()
{
#if JSC_OBJC_API_ENABLED
    if (!canExecuteScripts(ReasonForCallingCanExecuteScripts::NotAboutToExecuteScript))
        return 0;
    JSContext *context = [JSContext contextWithJSGlobalContextRef:toGlobalRef(bindingRootObject()->globalObject())];
    return context;
#else
    return 0;
#endif
}

void ScriptController::updatePlatformScriptObjects()
{
    if (m_windowScriptObject) {
        JSC::Bindings::RootObject* root = bindingRootObject();
        [m_windowScriptObject _setOriginRootObject:root andRootObject:root];
    }
}

void ScriptController::disconnectPlatformScriptObjects()
{
    if (m_windowScriptObject)
        disconnectWindowWrapper(m_windowScriptObject.get());
}

}
