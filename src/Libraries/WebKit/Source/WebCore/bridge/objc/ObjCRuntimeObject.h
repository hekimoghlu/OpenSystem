/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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
#ifndef ObjCRuntimeObject_h
#define ObjCRuntimeObject_h

#include "runtime_object.h"

namespace JSC {
namespace Bindings {

class ObjcInstance;

class ObjCRuntimeObject final : public RuntimeObject {
public:
    using Base = RuntimeObject;

    static ObjCRuntimeObject* create(VM& vm, Structure* structure, RefPtr<ObjcInstance>&& inst)
    {
        ObjCRuntimeObject* object = new (NotNull, allocateCell<ObjCRuntimeObject>(vm)) ObjCRuntimeObject(vm, structure, WTFMove(inst));
        object->finishCreation(vm);
        return object;
    }

    ObjcInstance* getInternalObjCInstance() const;

    DECLARE_INFO;

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
    {
        return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
    }

private:
    ObjCRuntimeObject(VM&, Structure*, RefPtr<ObjcInstance>&&);
    void finishCreation(VM&);
};

}
}

#endif
