/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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
#import "DOMInternal.h"

#import "DOMNodeInternal.h"
#import <WebCore/Document.h>
#import <WebCore/FrameDestructionObserverInlines.h>
#import <WebCore/JSNode.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/ScriptController.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <WebCore/runtime_root.h>
#import <wtf/HashMap.h>
#import <wtf/Lock.h>
#import <wtf/NeverDestroyed.h>

#if PLATFORM(IOS_FAMILY)
#define NEEDS_WRAPPER_CACHE_LOCK 1
#endif

//------------------------------------------------------------------------------------------
// Wrapping WebCore implementation objects

#ifdef NEEDS_WRAPPER_CACHE_LOCK
static Lock wrapperCacheLock;
static HashMap<DOMObjectInternal*, NSObject *>& wrapperCache() WTF_REQUIRES_LOCK(wrapperCacheLock)
#else
static HashMap<DOMObjectInternal*, NSObject *>& wrapperCache()
#endif
{
    static NeverDestroyed<HashMap<DOMObjectInternal*, NSObject *>> map;
    return map;
}

NSObject* getDOMWrapper(DOMObjectInternal* impl)
{
#ifdef NEEDS_WRAPPER_CACHE_LOCK
    Locker stateLocker { wrapperCacheLock };
#endif
    return wrapperCache().get(impl);
}

void addDOMWrapper(NSObject* wrapper, DOMObjectInternal* impl)
{
#ifdef NEEDS_WRAPPER_CACHE_LOCK
    Locker stateLocker { wrapperCacheLock };
#endif
    wrapperCache().set(impl, wrapper);
}

void removeDOMWrapper(DOMObjectInternal* impl)
{
#ifdef NEEDS_WRAPPER_CACHE_LOCK
    Locker stateLocker { wrapperCacheLock };
#endif
    wrapperCache().remove(impl);
}

//------------------------------------------------------------------------------------------

@implementation WebScriptObject (WebScriptObjectInternal)

// Only called by DOMObject subclass.
- (id)_init
{
    self = [super init];

    if (![self isKindOfClass:[DOMObject class]]) {
        [NSException raise:NSGenericException format:@"+%@: _init is an internal initializer", [self class]];
        return nil;
    }

    _private = [[WebScriptObjectPrivate alloc] init];
    _private->isCreatedByDOMWrapper = YES;
    
    return self;
}

- (void)_initializeScriptDOMNodeImp
{
    ASSERT(_private->isCreatedByDOMWrapper);
    
    if (![self isKindOfClass:[DOMNode class]]) {
        // DOMObject can't map back to a document, and thus an interpreter,
        // so for now only create wrappers for DOMNodes.
        return;
    }
    
    // Extract the WebCore::Node from the ObjectiveC wrapper.
    DOMNode *n = (DOMNode *)self;
    WebCore::Node *nodeImpl = core(n);

    // Dig up Interpreter and ExecState.
    auto* frame = nodeImpl->document().frame();
    if (!frame)
        return;

    // The global object which should own this node - FIXME: does this need to be isolated-world aware?
    WebCore::JSDOMGlobalObject* globalObject = frame->script().globalObject(WebCore::mainThreadNormalWorld());

    // Get (or create) a cached JS object for the DOM node.
    JSC::JSObject *scriptImp = asObject(WebCore::toJS(globalObject, globalObject, nodeImpl));

    JSC::Bindings::RootObject* rootObject = frame->script().bindingRootObject();

    [self _setImp:scriptImp originRootObject:rootObject rootObject:rootObject];
}

@end
