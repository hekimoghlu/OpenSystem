/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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
#ifndef KJS_BINDINGS_OBJC_CLASS_H
#define KJS_BINDINGS_OBJC_CLASS_H

#include "objc_runtime.h"
#include <wtf/text/WTFString.h>

namespace JSC {
namespace Bindings {

class ObjcClass : public Class
{
protected:
    ObjcClass (ClassStructPtr aClass); // Use classForIsA to create an ObjcClass.
    
public:
    // Return the cached ObjC of the specified name.
    static ObjcClass *classForIsA(ClassStructPtr);
    
    virtual Method* methodNamed(PropertyName, Instance*) const;
    virtual Field *fieldNamed(PropertyName, Instance*) const;

    virtual JSValue fallbackObject(JSGlobalObject*, Instance*, PropertyName);
    
    ClassStructPtr isa() { return _isa; }
    
private:
    ClassStructPtr _isa;
    mutable HashMap<String, std::unique_ptr<Method>> m_methodCache;
    mutable HashMap<String, std::unique_ptr<Field>> m_fieldCache;
};

} // namespace Bindings
} // namespace JSC

#endif
