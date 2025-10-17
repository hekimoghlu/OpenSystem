/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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

#include "Structure.h"
#include "TemplateObjectDescriptor.h"

namespace JSC {

class JSTemplateObjectDescriptor final : public JSCell {
public:
    using Base = JSCell;

    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;
    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.templateObjectDescriptorSpace<mode>();
    }
    DECLARE_INFO;

    static JSTemplateObjectDescriptor* create(VM&, Ref<TemplateObjectDescriptor>&&, int);

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    const TemplateObjectDescriptor& descriptor() const { return m_descriptor.get(); }

    JSArray* createTemplateObject(JSGlobalObject*);

    int endOffset() const { return m_endOffset; }

private:
    JSTemplateObjectDescriptor(VM&, Ref<TemplateObjectDescriptor>&&, int);

    static void destroy(JSCell*);

    Ref<TemplateObjectDescriptor> m_descriptor;
    int m_endOffset { 0 };
};

} // namespace JSC
