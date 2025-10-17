/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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

#include "CachedScriptSourceProvider.h"
#include "DOMPromiseProxy.h"
#include "ScriptBufferSourceProvider.h"
#include "WebAssemblyCachedScriptSourceProvider.h"
#include "WebAssemblyScriptBufferSourceProvider.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CachedScriptSourceProvider);
WTF_MAKE_TZONE_ALLOCATED_IMPL(DOMPromiseDeferredBase);
WTF_MAKE_TZONE_ALLOCATED_IMPL(ScriptBufferSourceProvider);
#if ENABLE(WEBASSEMBLY)
WTF_MAKE_TZONE_ALLOCATED_IMPL(WebAssemblyCachedScriptSourceProvider);
WTF_MAKE_TZONE_ALLOCATED_IMPL(WebAssemblyScriptBufferSourceProvider);
#endif

#define TZONE_TEMPLATE_PARAMS
#define TZONE_TYPE DOMPromiseProxy<IDLUndefined>

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL_WITH_MULTIPLE_OR_SPECIALIZED_PARAMETERS();

#undef TZONE_TEMPLATE_PARAMS
#undef TZONE_TYPE

} // namespace WebCore
