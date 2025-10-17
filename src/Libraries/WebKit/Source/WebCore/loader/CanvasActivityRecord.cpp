/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#include "CanvasActivityRecord.h"

const unsigned maximumNumberOfStringsToRecord = 10;
namespace WebCore {
bool CanvasActivityRecord::recordWrittenOrMeasuredText(const String& text)
{
    // We limit the size of the textWritten HashSet to save memory and prevent bloating
    // the plist with the resourceLoadStatistics entries. A few strings is often enough
    // to provide sufficient information about the state of canvas activity.
    if (textWritten.size() >= maximumNumberOfStringsToRecord)
        return false;
    return textWritten.add(text).isNewEntry;
}

void CanvasActivityRecord::mergeWith(const CanvasActivityRecord& otherCanvasActivityRecord)
{
    textWritten.add(otherCanvasActivityRecord.textWritten.begin(), otherCanvasActivityRecord.textWritten.end());
    wasDataRead |= otherCanvasActivityRecord.wasDataRead;
}
} // namespace WebCore
