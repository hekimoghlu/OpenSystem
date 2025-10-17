/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 11, 2023.
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
#import "TextChecking.h"

#if PLATFORM(MAC)

#import <Foundation/Foundation.h>

namespace WebCore {

NSTextCheckingTypes nsTextCheckingTypes(OptionSet<TextCheckingType> types)
{
    NSTextCheckingTypes mask = 0;
    if (types.contains(TextCheckingType::Spelling))
        mask |= NSTextCheckingTypeSpelling;
    if (types.contains(TextCheckingType::Grammar))
        mask |= NSTextCheckingTypeGrammar;
    if (types.contains(TextCheckingType::Link))
        mask |= NSTextCheckingTypeLink;
    if (types.contains(TextCheckingType::Quote))
        mask |= NSTextCheckingTypeQuote;
    if (types.contains(TextCheckingType::Dash))
        mask |= NSTextCheckingTypeDash;
    if (types.contains(TextCheckingType::Replacement))
        mask |= NSTextCheckingTypeReplacement;
    if (types.contains(TextCheckingType::Correction))
        mask |= NSTextCheckingTypeCorrection;
    return mask;
}

} // namespace WebCore

#endif // PLATFORM(MAC)
