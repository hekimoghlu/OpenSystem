/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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
#include "GCRequest.h"

namespace JSC {

bool GCRequest::subsumedBy(const GCRequest& other) const
{
    // If we have callbacks, then there is no chance that we're subsumed by an existing request.
    if (didFinishEndPhase)
        return false;
    
    if (other.scope == CollectionScope::Full)
        return true;
    
    if (scope) {
        // If we're eden, then we're subsumed by the other scope because the other scope is either eden
        // or disengaged (so either eden or full). If we're full, then we're not subsumed, for the same
        // reason.
        return scope == CollectionScope::Eden;
    }
    
    // At this point we know that other.scope is either not engaged or Eden, and this.scope is not
    // engaged. So, we're expecting to do either an eden or full collection, and the other scope is
    // either the same or is requesting specifically a full collection. We are subsumed if the other
    // scope is disengaged (so same as us).
    return !other.scope;
}

void GCRequest::dump(PrintStream& out) const
{
    out.print("{scope = ", scope, ", didFinishEndPhase = ", didFinishEndPhase ? "engaged" : "null", "}");
}

} // namespace JSC

