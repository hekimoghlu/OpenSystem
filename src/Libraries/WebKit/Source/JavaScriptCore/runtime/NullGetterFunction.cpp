/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
#include "NullGetterFunction.h"

#include "JSCInlines.h"

namespace JSC {

const ClassInfo NullGetterFunction::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(NullGetterFunction) };

namespace NullGetterFunctionInternal {

static JSC_DECLARE_HOST_FUNCTION(callReturnUndefined);

JSC_DEFINE_HOST_FUNCTION(callReturnUndefined, (JSGlobalObject*, CallFrame*))
{
    return JSValue::encode(jsUndefined());
}
}

NullGetterFunction::NullGetterFunction(VM& vm, Structure* structure)
    : Base(vm, structure, NullGetterFunctionInternal::callReturnUndefined, nullptr)
{
}

}
