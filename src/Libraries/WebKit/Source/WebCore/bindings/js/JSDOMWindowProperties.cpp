/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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
#include "JSDOMWindowProperties.h"

#include "HTMLDocument.h"
#include "JSDOMBinding.h"
#include "JSDOMBindingSecurity.h"
#include "JSDOMConvertStrings.h"
#include "JSDOMWindowBase.h"
#include "JSElement.h"
#include "JSHTMLCollection.h"
#include "LocalFrame.h"
#include "WebCoreJSClientData.h"

namespace WebCore {

using namespace JSC;

const ClassInfo JSDOMWindowProperties::s_info = { "WindowProperties"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSDOMWindowProperties) };

// https://html.spec.whatwg.org/multipage/window-object.html#dom-window-nameditem
static bool jsDOMWindowPropertiesGetOwnPropertySlotNamedItemGetter(JSDOMWindowProperties* thisObject, LocalDOMWindow& window, JSGlobalObject* lexicalGlobalObject, PropertyName propertyName, PropertySlot& slot)
{
    if (auto* frame = window.frame()) {
        if (auto* scopedChild = dynamicDowncast<LocalFrame>(frame->tree().scopedChildBySpecifiedName(propertyNameToAtomString(propertyName)))) {
            slot.setValue(thisObject, enumToUnderlyingType(PropertyAttribute::DontEnum), toJS(lexicalGlobalObject, scopedChild->document()->domWindow()));
            return true;
        }
    }

    if (!BindingSecurity::shouldAllowAccessToDOMWindow(lexicalGlobalObject, window, ThrowSecurityError))
        return false;

    // Allow shortcuts like 'Image1' instead of document.images.Image1
    auto* document = window.document();
    if (auto* htmlDocument = dynamicDowncast<HTMLDocument>(document)) {
        AtomString atomPropertyName = propertyName.publicName();
        if (!atomPropertyName.isEmpty() && htmlDocument->hasWindowNamedItem(atomPropertyName)) {
            JSValue namedItem;
            if (UNLIKELY(htmlDocument->windowNamedItemContainsMultipleElements(atomPropertyName))) {
                Ref<HTMLCollection> collection = document->windowNamedItems(atomPropertyName);
                ASSERT(collection->length() > 1);
                namedItem = toJS(lexicalGlobalObject, thisObject->globalObject(), collection);
            } else
                namedItem = toJS(lexicalGlobalObject, thisObject->globalObject(), htmlDocument->windowNamedItem(atomPropertyName).get());
            slot.setValue(thisObject, enumToUnderlyingType(PropertyAttribute::DontEnum), namedItem);
            return true;
        }
    }

    return false;
}

void JSDOMWindowProperties::finishCreation(JSGlobalObject& globalObject)
{
    VM& vm = globalObject.vm();
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
    JSObject::preventExtensions(this, &globalObject);
}

bool JSDOMWindowProperties::getOwnPropertySlot(JSObject* object, JSGlobalObject* lexicalGlobalObject, PropertyName propertyName, PropertySlot& slot)
{
    auto* thisObject = jsCast<JSDOMWindowProperties*>(object);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());

    if (Base::getOwnPropertySlot(thisObject, lexicalGlobalObject, propertyName, slot))
        return true;
    JSObject* proto = asObject(thisObject->getPrototypeDirect());
    if (proto->hasProperty(lexicalGlobalObject, propertyName))
        return false;

    auto* jsWindow = jsDynamicCast<JSDOMWindowBase*>(thisObject->globalObject());
    if (!jsWindow)
        return false;

    auto& window = jsWindow->wrapped();
    RefPtr localWindow = dynamicDowncast<LocalDOMWindow>(window);
    if (!localWindow)
        return false;
    return jsDOMWindowPropertiesGetOwnPropertySlotNamedItemGetter(thisObject, *localWindow, lexicalGlobalObject, propertyName, slot);
}

bool JSDOMWindowProperties::getOwnPropertySlotByIndex(JSObject* object, JSGlobalObject* lexicalGlobalObject, unsigned index, PropertySlot& slot)
{
    VM& vm = lexicalGlobalObject->vm();
    return getOwnPropertySlot(object, lexicalGlobalObject, Identifier::from(vm, index), slot);
}

bool JSDOMWindowProperties::deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&)
{
    return false;
}

bool JSDOMWindowProperties::deletePropertyByIndex(JSCell*, JSGlobalObject*, unsigned)
{
    return false;
}

bool JSDOMWindowProperties::preventExtensions(JSObject*, JSGlobalObject*)
{
    return false;
}

bool JSDOMWindowProperties::isExtensible(JSObject*, JSGlobalObject*)
{
    return true;
}

bool JSDOMWindowProperties::defineOwnProperty(JSObject*, JSGlobalObject* lexicalGlobalObject, PropertyName, const PropertyDescriptor&, bool shouldThrow)
{
    auto scope = DECLARE_THROW_SCOPE(lexicalGlobalObject->vm());
    return typeError(lexicalGlobalObject, scope, shouldThrow, "Defining a property on a WindowProperties object is not allowed."_s);
}

JSC::GCClient::IsoSubspace* JSDOMWindowProperties::subspaceForImpl(JSC::VM& vm)
{
    return &static_cast<JSVMClientData*>(vm.clientData)->domWindowPropertiesSpace();
}

} // namespace WebCore
