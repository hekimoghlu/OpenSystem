/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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
#import "AlternativeTextUIController.h"

#import "FloatRect.h"
#import <wtf/TZoneMallocInlines.h>

#if USE(APPKIT)
#import <AppKit/NSSpellChecker.h>
#import <AppKit/NSTextAlternatives.h>
#import <AppKit/NSView.h>
#endif

#if PLATFORM(IOS_FAMILY)
#import <pal/spi/ios/UIKitSPI.h>
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AlternativeTextUIController);

std::optional<DictationContext> AlternativeTextUIController::addAlternatives(PlatformTextAlternatives *alternatives)
{
    return m_contextController.addAlternatives(alternatives);
}

void AlternativeTextUIController::replaceAlternatives(PlatformTextAlternatives *alternatives, DictationContext context)
{
    m_contextController.replaceAlternatives(alternatives, context);
}

PlatformTextAlternatives *AlternativeTextUIController::alternativesForContext(DictationContext context)
{
    return m_contextController.alternativesForContext(context);
}

void AlternativeTextUIController::clear()
{
    return m_contextController.clear();
}

#if USE(APPKIT)

void AlternativeTextUIController::showAlternatives(NSView *view, const FloatRect& boundingBoxOfPrimaryString, DictationContext context, AcceptanceHandler acceptanceHandler)
{
    dismissAlternatives();
    if (!view)
        return;

    m_view = view;

    PlatformTextAlternatives *alternatives = m_contextController.alternativesForContext(context);
    if (!alternatives)
        return;

    [[NSSpellChecker sharedSpellChecker] showCorrectionIndicatorOfType:NSCorrectionIndicatorTypeGuesses primaryString:alternatives.primaryString alternativeStrings:alternatives.alternativeStrings forStringInRect:boundingBoxOfPrimaryString view:m_view.get() completionHandler:^(NSString *acceptedString) {
        if (acceptedString) {
            handleAcceptedAlternative(acceptedString, context, alternatives);
            acceptanceHandler(acceptedString);
        }
    }];
}

void AlternativeTextUIController::handleAcceptedAlternative(NSString *acceptedAlternative, DictationContext context, PlatformTextAlternatives *alternatives)
{
    [alternatives noteSelectedAlternativeString:acceptedAlternative];
    m_contextController.removeAlternativesForContext(context);
    m_view = nullptr;
}

void AlternativeTextUIController::dismissAlternatives()
{
    if (m_view)
        [[NSSpellChecker sharedSpellChecker] dismissCorrectionIndicatorForView:m_view.get()];
}

#endif

void AlternativeTextUIController::removeAlternatives(DictationContext context)
{
    m_contextController.removeAlternativesForContext(context);
}

} // namespace WebCore
