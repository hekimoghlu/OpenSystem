/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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

#import "JSDOMBinding.h"
#import "ObjCRuntimeObject.h"
#import "objc_instance.h"
#import <JavaScriptCore/ObjectPrototype.h>

namespace JSC {
namespace Bindings {

const ClassInfo ObjCRuntimeObject::s_info = { "ObjCRuntimeObject"_s, &RuntimeObject::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(ObjCRuntimeObject) };

ObjCRuntimeObject::ObjCRuntimeObject(VM& vm, Structure* structure, RefPtr<ObjcInstance>&& instance)
    : RuntimeObject(vm, structure, instance)
{
}

void ObjCRuntimeObject::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(&s_info));
}

ObjcInstance* ObjCRuntimeObject::getInternalObjCInstance() const
{
    return static_cast<ObjcInstance*>(getInternalInstance());
}


}
}
