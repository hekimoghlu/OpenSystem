/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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
#import "DictationContext.h"
#import "PlatformTextAlternatives.h"
#import <wtf/HashMap.h>
#import <wtf/RetainPtr.h>

namespace WebCore {

class AlternativeTextContextController {
public:
    std::optional<DictationContext> addAlternatives(PlatformTextAlternatives *);
    void replaceAlternatives(PlatformTextAlternatives *, DictationContext);
    void removeAlternativesForContext(DictationContext);
    void clear();

    PlatformTextAlternatives *alternativesForContext(DictationContext) const;

private:
    UncheckedKeyHashMap<DictationContext, RetainPtr<PlatformTextAlternatives>> m_alternatives;
    UncheckedKeyHashMap<RetainPtr<PlatformTextAlternatives>, DictationContext> m_contexts;
};

} // namespace WebCore
