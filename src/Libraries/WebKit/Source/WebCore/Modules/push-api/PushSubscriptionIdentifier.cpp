/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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
#include "PushSubscriptionIdentifier.h"

#include <wtf/text/StringBuilder.h>

namespace WebCore {

// Do not change this without thinking about backwards compatibility. Topics are persisted in both PushDatabase and APS.
String makePushTopic(const PushSubscriptionSetIdentifier& set, const String& scope)
{
    StringBuilder builder;
    builder.reserveCapacity(
        set.bundleIdentifier.length() +
        (!set.pushPartition.isEmpty() ? 6 + set.pushPartition.length() : 0) +
        (set.dataStoreIdentifier ? 40 : 0) +
        1 + scope.length());
    builder.append(set.bundleIdentifier);
    if (!set.pushPartition.isEmpty())
        builder.append(" part:"_s, set.pushPartition);
    if (set.dataStoreIdentifier)
        builder.append(" ds:"_s, set.dataStoreIdentifier->toString());
    builder.append(" "_s, scope);
    return builder.toString();
}

String PushSubscriptionSetIdentifier::debugDescription() const
{
    StringBuilder builder;
    builder.reserveCapacity(
        1 + bundleIdentifier.length() +
        (!pushPartition.isEmpty() ? 6 + pushPartition.length() : 0) +
        (dataStoreIdentifier ? 12 : 0) + 1);
    builder.append('[', bundleIdentifier);
    if (!pushPartition.isEmpty())
        builder.append(" part:"_s, pushPartition);
    if (dataStoreIdentifier)
        builder.append(" ds:"_s, dataStoreIdentifier->toString(), 0, 8);
    builder.append(']');
    return builder.toString();
}

} // namespace WebCore
