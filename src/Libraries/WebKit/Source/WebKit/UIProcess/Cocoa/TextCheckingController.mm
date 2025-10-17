/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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
#import "TextCheckingController.h"

#if ENABLE(PLATFORM_DRIVEN_TEXT_CHECKING)

#import "MessageSenderInlines.h"
#import "TextCheckingControllerProxyMessages.h"
#import "WebPageProxy.h"
#import "WebProcessProxy.h"
#import <WebCore/AttributedString.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextCheckingController);

TextCheckingController::TextCheckingController(WebPageProxy& webPageProxy)
    : m_page(webPageProxy)
{
}

void TextCheckingController::replaceRelativeToSelection(NSAttributedString *annotatedString, int64_t selectionOffset, uint64_t length, uint64_t relativeReplacementLocation, uint64_t relativeReplacementLength)
{
    if (!m_page->hasRunningProcess())
        return;

    m_page->legacyMainFrameProcess().send(Messages::TextCheckingControllerProxy::ReplaceRelativeToSelection(WebCore::AttributedString::fromNSAttributedString(annotatedString), selectionOffset, length, relativeReplacementLocation, relativeReplacementLength), m_page->webPageIDInMainFrameProcess());
}

void TextCheckingController::removeAnnotationRelativeToSelection(NSString *annotationName, int64_t selectionOffset, uint64_t length)
{
    if (!m_page->hasRunningProcess())
        return;

    m_page->legacyMainFrameProcess().send(Messages::TextCheckingControllerProxy::RemoveAnnotationRelativeToSelection(annotationName, selectionOffset, length), m_page->webPageIDInMainFrameProcess());
}

} // namespace WebKit

#endif // ENABLE(PLATFORM_DRIVEN_TEXT_CHECKING)
