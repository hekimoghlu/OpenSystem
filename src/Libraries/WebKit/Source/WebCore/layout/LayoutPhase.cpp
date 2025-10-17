/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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
#include "LayoutPhase.h"

#include <wtf/NeverDestroyed.h>

namespace WebCore {
namespace Layout {

static Phase& phase()
{
    static NeverDestroyed<Phase> phase;
    return phase;
}

bool Phase::isInTreeBuilding()
{ 
    return *phase() && (*phase()).value() == Type::TreeBuilding;
}

bool Phase::isInLayout()
{ 
    return *phase() && (*phase()).value() == Type::Layout;
}

bool Phase::isInInvalidation()
{ 
    return *phase() && (*phase()).value() == Type::Invalidation;
}

PhaseScope::PhaseScope(Phase::Type type)
{ 
    // Should never nest states like calling TreeBuilding from Layout. 
    ASSERT(!(*phase()).has_value());
    phase().set(type);
}
    
PhaseScope::~PhaseScope()
{
    phase().reset();
}

}
}
