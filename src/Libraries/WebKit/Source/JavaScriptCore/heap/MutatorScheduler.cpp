/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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
#include "MutatorScheduler.h"

#include <wtf/TZoneMallocInlines.h>
#include <wtf/TimeWithDynamicClockType.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MutatorScheduler);

MutatorScheduler::MutatorScheduler() = default;

MutatorScheduler::~MutatorScheduler() = default;

void MutatorScheduler::didStop()
{
}

void MutatorScheduler::willResume()
{
}

void MutatorScheduler::didReachTermination()
{
}

void MutatorScheduler::didExecuteConstraints()
{
}

void MutatorScheduler::synchronousDrainingDidStall()
{
}

void MutatorScheduler::log()
{
}

bool MutatorScheduler::shouldStop()
{
    return hasElapsed(timeToStop());
}

bool MutatorScheduler::shouldResume()
{
    return hasElapsed(timeToResume());
}

} // namespace JSC

