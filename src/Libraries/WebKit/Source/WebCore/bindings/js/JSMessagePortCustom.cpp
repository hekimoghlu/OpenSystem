/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
#include "JSMessagePort.h"
#include "WebCoreOpaqueRootInlines.h"

namespace WebCore {
using namespace JSC;

template<typename Visitor>
void JSMessagePort::visitAdditionalChildren(Visitor& visitor)
{
    // If we have a locally entangled port, we can directly mark it as reachable. Ports that are remotely entangled are marked in-use by markActiveObjectsForContext().
    if (auto* port = wrapped().locallyEntangledPort())
        addWebCoreOpaqueRoot(visitor, *port);
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSMessagePort);

} // namespace WebCore
