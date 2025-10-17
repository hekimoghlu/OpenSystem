/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 8, 2024.
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
#import "LocalCurrentTraitCollection.h"

#if PLATFORM(IOS_FAMILY)

#import <pal/ios/UIKitSoftLink.h>

namespace WebCore {

UITraitCollection *traitCollectionWithAdjustedIdiomForSystemColors(UITraitCollection *traitCollection)
{
#if PLATFORM(VISION)
    // Use the iPad idiom instead of the Vision idiom, since some system colors are transparent
    // in the Vision idiom, and are not web-compatible.
    if (traitCollection.userInterfaceIdiom == UIUserInterfaceIdiomVision) {
        return [traitCollection traitCollectionByModifyingTraits:^(id<UIMutableTraits> traits) {
            traits.userInterfaceIdiom = UIUserInterfaceIdiomPad;
        }];
    }
#endif
    return traitCollection;
}

LocalCurrentTraitCollection::LocalCurrentTraitCollection(bool useDarkAppearance, bool useElevatedUserInterfaceLevel)
{
    m_savedTraitCollection = [PAL::getUITraitCollectionClass() currentTraitCollection];

    // FIXME: <rdar://problem/96607991> `-[UITraitCollection currentTraitCollection]` is not guaranteed
    // to return a useful set of traits in cases where it has not been explicitly set. Ideally, this
    // method should also take in a base, full-specified trait collection from the view hierarchy, to be
    // used when building the new trait collection.
    RetainPtr combinedTraits = [m_savedTraitCollection traitCollectionByModifyingTraits:^(id<UIMutableTraits> traits) {
        traits.userInterfaceStyle = useDarkAppearance ? UIUserInterfaceStyleDark : UIUserInterfaceStyleLight;
        traits.userInterfaceLevel = useElevatedUserInterfaceLevel ? UIUserInterfaceLevelElevated : UIUserInterfaceLevelBase;
    }];

    [PAL::getUITraitCollectionClass() setCurrentTraitCollection:traitCollectionWithAdjustedIdiomForSystemColors(combinedTraits.get())];
}

LocalCurrentTraitCollection::LocalCurrentTraitCollection(UITraitCollection *traitCollection)
{
    m_savedTraitCollection = [PAL::getUITraitCollectionClass() currentTraitCollection];
    auto newTraitCollection = traitCollectionWithAdjustedIdiomForSystemColors(traitCollection);
    [PAL::getUITraitCollectionClass() setCurrentTraitCollection:newTraitCollection];
}

LocalCurrentTraitCollection::~LocalCurrentTraitCollection()
{
    [PAL::getUITraitCollectionClass() setCurrentTraitCollection:m_savedTraitCollection.get()];
}

}

#endif // PLATFORM(IOS_FAMILY)
