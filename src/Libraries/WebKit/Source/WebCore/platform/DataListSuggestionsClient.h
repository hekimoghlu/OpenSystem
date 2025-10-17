/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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
#pragma once

#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>

namespace WebCore {

class IntRect;
struct DataListSuggestion;

class DataListSuggestionsClient : public CanMakeCheckedPtr<DataListSuggestionsClient> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DataListSuggestionsClient);
public:
    virtual ~DataListSuggestionsClient() = default;

    virtual IntRect elementRectInRootViewCoordinates() const = 0;
    virtual Vector<DataListSuggestion> suggestions() = 0;

    virtual void didSelectDataListOption(const String&) = 0;
    virtual void didCloseSuggestions() = 0;
};

}
