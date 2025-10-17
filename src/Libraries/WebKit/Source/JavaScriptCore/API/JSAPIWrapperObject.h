/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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
#ifndef JSAPIWrapperObject_h
#define JSAPIWrapperObject_h

#include "JSBase.h"
#include "JSDestructibleObject.h"

#if JSC_OBJC_API_ENABLED || defined(JSC_GLIB_API_ENABLED)

namespace JSC {
    
class JSAPIWrapperObject : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    template<typename, SubspaceAccess>
    static void subspaceFor(VM&) { RELEASE_ASSERT_NOT_REACHED(); }
    
    void finishCreation(VM&);
    DECLARE_VISIT_CHILDREN_WITH_MODIFIER(JS_EXPORT_PRIVATE);
    
    void* wrappedObject() { return m_wrappedObject; }
    void setWrappedObject(void*);

protected:
    JSAPIWrapperObject(VM&, Structure*);

private:
    void* m_wrappedObject { nullptr };
};

} // namespace JSC

#endif // JSC_OBJC_API_ENABLED || defined(JSC_GLIB_API_ENABLED)

#endif // JSAPIWrapperObject_h
