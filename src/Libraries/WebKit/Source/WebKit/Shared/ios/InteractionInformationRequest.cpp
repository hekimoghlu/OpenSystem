/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 22, 2022.
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
#import "config.h"
#import "InteractionInformationRequest.h"


namespace WebKit {

#if PLATFORM(IOS_FAMILY)

bool InteractionInformationRequest::isValidForRequest(const InteractionInformationRequest& other, int radius) const
{
    if (other.includeSnapshot && !includeSnapshot)
        return false;

    if (other.includeLinkIndicator && !includeLinkIndicator)
        return false;

    if (other.includeCursorContext && !includeCursorContext)
        return false;

    if (other.includeHasDoubleClickHandler && !includeHasDoubleClickHandler)
        return false;

    if (other.includeImageData && !includeImageData)
        return false;

    if (other.gatherAnimations && !gatherAnimations)
        return false;

    if (other.linkIndicatorShouldHaveLegacyMargins != linkIndicatorShouldHaveLegacyMargins)
        return false;

    return (other.point - point).diagonalLengthSquared() <= radius * radius;
}
    
bool InteractionInformationRequest::isApproximatelyValidForRequest(const InteractionInformationRequest& other, int radius) const
{
    return isValidForRequest(other, radius);
}

#endif // PLATFORM(IOS_FAMILY)

}
