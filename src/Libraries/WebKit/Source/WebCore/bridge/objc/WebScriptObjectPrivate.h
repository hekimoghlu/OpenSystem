/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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
#import "WebScriptObject.h"
#import <JavaScriptCore/JSCJSValue.h>
#import <wtf/RefPtr.h>

namespace JSC {
    class JSObject;
    namespace Bindings {
        class RootObject;
    }
}

namespace WebCore {
    using CreateWrapperFunction = WebScriptObject * (*)(JSC::JSObject&);
    using DisconnectWindowWrapperFunction = void (*)(WebScriptObject *);
    WEBCORE_EXPORT void initializeDOMWrapperHooks(CreateWrapperFunction, DisconnectWindowWrapperFunction);

    NSObject *getJSWrapper(JSC::JSObject*);
    void addJSWrapper(NSObject *wrapper, JSC::JSObject*);
    void removeJSWrapper(JSC::JSObject*);
    id createJSWrapper(JSC::JSObject*, RefPtr<JSC::Bindings::RootObject>&& origin, RefPtr<JSC::Bindings::RootObject>&&);

    void disconnectWindowWrapper(WebScriptObject *);
}

@interface WebScriptObject (Private)
+ (id)_convertValueToObjcValue:(JSC::JSValue)value originRootObject:(JSC::Bindings::RootObject*)originRootObject rootObject:(JSC::Bindings::RootObject*)rootObject;
+ (id)scriptObjectForJSObject:(JSObjectRef)jsObject originRootObject:(JSC::Bindings::RootObject*)originRootObject rootObject:(JSC::Bindings::RootObject*)rootObject;
- (id)_init;
- (id)_initWithJSObject:(JSC::JSObject*)imp originRootObject:(RefPtr<JSC::Bindings::RootObject>&&)originRootObject rootObject:(RefPtr<JSC::Bindings::RootObject>&&)rootObject;
- (void)_setImp:(JSC::JSObject*)imp originRootObject:(RefPtr<JSC::Bindings::RootObject>&&)originRootObject rootObject:(RefPtr<JSC::Bindings::RootObject>&&)rootObject;
- (void)_setOriginRootObject:(RefPtr<JSC::Bindings::RootObject>&&)originRootObject andRootObject:(RefPtr<JSC::Bindings::RootObject>&&)rootObject;
- (void)_initializeScriptDOMNodeImp;
- (JSC::JSObject*)_imp;
- (BOOL)_hasImp;
- (JSC::Bindings::RootObject*)_rootObject;
- (JSC::Bindings::RootObject*)_originRootObject;
- (JSGlobalContextRef)_globalContextRef;
@end

@interface WebScriptObject (StagedForPublic)
/*!
 @method hasWebScriptKey:
 @param name The name of the property to check for.
 @discussion Checks for the existence of the property on the object in the script environment.
 @result Returns YES if the property exists, NO otherwise.
 */
- (BOOL)hasWebScriptKey:(NSString *)name;
@end

WEBCORE_EXPORT @interface WebScriptObjectPrivate : NSObject
{
@public
    JSC::JSObject* imp;
    JSC::Bindings::RootObject* rootObject;
    JSC::Bindings::RootObject* originRootObject;
    BOOL isCreatedByDOMWrapper;
}
@end
