/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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
#include "WebFindOptions.h"

namespace WebKit {

WebCore::FindOptions core(OptionSet<FindOptions> options)
{
    WebCore::FindOptions result;
    if (options.contains(FindOptions::CaseInsensitive))
        result.add(WebCore::FindOption::CaseInsensitive);
    if (options.contains(FindOptions::AtWordStarts))
        result.add(WebCore::FindOption::AtWordStarts);
    if (options.contains(FindOptions::TreatMedialCapitalAsWordStart))
        result.add(WebCore::FindOption::TreatMedialCapitalAsWordStart);
    if (options.contains(FindOptions::Backwards))
        result.add(WebCore::FindOption::Backwards);
    if (options.contains(FindOptions::WrapAround))
        result.add(WebCore::FindOption::WrapAround);
    if (options.contains(FindOptions::AtWordEnds))
        result.add(WebCore::FindOption::AtWordEnds);
    if (options.contains(FindOptions::DoNotSetSelection))
        result.add(WebCore::FindOption::DoNotSetSelection);
    return result;
}

} // namespace WebKit
