/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 30, 2023.
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
#include "JSCVirtualMachine.h"

#include "JSCContextPrivate.h"
#include "JSCVirtualMachinePrivate.h"
#include "JSContextRef.h"
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/glib/WTFGType.h>

/**
 * JSCVirtualMachine:
 * @short_description: JavaScript Virtual Machine
 * @title: JSCVirtualMachine
 * @see_also: JSCContext
 *
 * JSCVirtualMachine represents a group of JSCContext<!-- -->s. It allows
 * concurrent JavaScript execution by creating a different instance of
 * JSCVirtualMachine in each thread.
 *
 * To create a group of JSCContext<!-- -->s pass the same JSCVirtualMachine
 * instance to every JSCContext constructor.
 */

struct _JSCVirtualMachinePrivate {
    JSContextGroupRef jsContextGroup;
    UncheckedKeyHashMap<JSGlobalContextRef, JSCContext*> contextCache;
};

WEBKIT_DEFINE_FINAL_TYPE(JSCVirtualMachine, jsc_virtual_machine, G_TYPE_OBJECT, GObject)

static Lock wrapperCacheMutex;

static UncheckedKeyHashMap<JSContextGroupRef, JSCVirtualMachine*>& wrapperMap() WTF_REQUIRES_LOCK(wrapperCacheMutex)
{
    static LazyNeverDestroyed<UncheckedKeyHashMap<JSContextGroupRef, JSCVirtualMachine*>> shared;
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        shared.construct();
    });
    return shared;
}

static void addWrapper(JSContextGroupRef group, JSCVirtualMachine* vm)
{
    Locker locker { wrapperCacheMutex };
    ASSERT(!wrapperMap().contains(group));
    wrapperMap().set(group, vm);
}

static void removeWrapper(JSContextGroupRef group)
{
    Locker locker { wrapperCacheMutex };
    ASSERT(wrapperMap().contains(group));
    wrapperMap().remove(group);
}

static void jscVirtualMachineSetContextGroup(JSCVirtualMachine *vm, JSContextGroupRef group)
{
    if (group) {
        ASSERT(!vm->priv->jsContextGroup);
        vm->priv->jsContextGroup = group;
        JSContextGroupRetain(vm->priv->jsContextGroup);
        addWrapper(vm->priv->jsContextGroup, vm);
    } else if (vm->priv->jsContextGroup) {
        removeWrapper(vm->priv->jsContextGroup);
        JSContextGroupRelease(vm->priv->jsContextGroup);
        vm->priv->jsContextGroup = nullptr;
    }
}

static void jscVirtualMachineEnsureContextGroup(JSCVirtualMachine *vm)
{
    if (vm->priv->jsContextGroup)
        return;

    auto* jsContextGroup = JSContextGroupCreate();
    jscVirtualMachineSetContextGroup(vm, jsContextGroup);
    JSContextGroupRelease(jsContextGroup);
}

static void jscVirtualMachineDispose(GObject* object)
{
    JSCVirtualMachine* vm = JSC_VIRTUAL_MACHINE(object);
    jscVirtualMachineSetContextGroup(vm, nullptr);

    G_OBJECT_CLASS(jsc_virtual_machine_parent_class)->dispose(object);
}

static void jsc_virtual_machine_class_init(JSCVirtualMachineClass* klass)
{
    GObjectClass* objClass = G_OBJECT_CLASS(klass);
    objClass->dispose = jscVirtualMachineDispose;
}

GRefPtr<JSCVirtualMachine> jscVirtualMachineGetOrCreate(JSContextGroupRef jsContextGroup)
{
    GRefPtr<JSCVirtualMachine> vm = wrapperMap().get(jsContextGroup);
    if (!vm) {
        vm = adoptGRef(jsc_virtual_machine_new());
        jscVirtualMachineSetContextGroup(vm.get(), jsContextGroup);
    }
    return vm;
}

JSContextGroupRef jscVirtualMachineGetContextGroup(JSCVirtualMachine* vm)
{
    jscVirtualMachineEnsureContextGroup(vm);
    return vm->priv->jsContextGroup;
}

void jscVirtualMachineAddContext(JSCVirtualMachine* vm, JSCContext* context)
{
    ASSERT(vm->priv->jsContextGroup);
    auto jsContext = jscContextGetJSContext(context);
    ASSERT(JSContextGetGroup(jsContext) == vm->priv->jsContextGroup);
    ASSERT(!vm->priv->contextCache.contains(jsContext));
    vm->priv->contextCache.set(jsContext, context);
}

void jscVirtualMachineRemoveContext(JSCVirtualMachine* vm, JSCContext* context)
{
    ASSERT(vm->priv->jsContextGroup);
    auto jsContext = jscContextGetJSContext(context);
    ASSERT(JSContextGetGroup(jsContext) == vm->priv->jsContextGroup);
    ASSERT(vm->priv->contextCache.contains(jsContext));
    vm->priv->contextCache.remove(jsContext);
}

JSCContext* jscVirtualMachineGetContext(JSCVirtualMachine* vm, JSGlobalContextRef jsContext)
{
    return vm->priv->contextCache.get(jsContext);
}

/**
 * jsc_virtual_machine_new:
 *
 * Create a new #JSCVirtualMachine.
 *
 * Returns: (transfer full): the newly created #JSCVirtualMachine.
 */
JSCVirtualMachine* jsc_virtual_machine_new()
{
    return JSC_VIRTUAL_MACHINE(g_object_new(JSC_TYPE_VIRTUAL_MACHINE, nullptr));
}
