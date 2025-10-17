/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 15, 2023.
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

#include <wtf/Noncopyable.h>
#include <wtf/RetainPtr.h>

#if PLATFORM(IOS_FAMILY)

OBJC_CLASS UITraitCollection;

namespace WebCore {

// This class automatically saves and restores the current UITraitCollection for
// functions which call out into UIKit and rely on the current UITraitCollection being set
class LocalCurrentTraitCollection {
    WTF_MAKE_NONCOPYABLE(LocalCurrentTraitCollection);

public:
    WEBCORE_EXPORT LocalCurrentTraitCollection(bool useDarkAppearance, bool useElevatedUserInterfaceLevel);
    WEBCORE_EXPORT LocalCurrentTraitCollection(UITraitCollection *);
    WEBCORE_EXPORT ~LocalCurrentTraitCollection();

private:
    RetainPtr<UITraitCollection> m_savedTraitCollection;
};

WEBCORE_EXPORT UITraitCollection *traitCollectionWithAdjustedIdiomForSystemColors(UITraitCollection *);

}

#endif // PLATFORM(IOS_FAMILY)

