/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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
#include "WebOpenPanelResultListener.h"

#include <WebCore/FileChooser.h>
#include <WebCore/Icon.h>

namespace WebKit {

Ref<WebOpenPanelResultListener> WebOpenPanelResultListener::create(WebPage& page, Ref<WebCore::FileChooser>&& fileChooser)
{
    return adoptRef(*new WebOpenPanelResultListener(page, WTFMove(fileChooser)));
}

WebOpenPanelResultListener::WebOpenPanelResultListener(WebPage& page, Ref<WebCore::FileChooser>&& fileChooser)
    : m_page(&page)
    , m_fileChooser(WTFMove(fileChooser))
{
}

WebOpenPanelResultListener::~WebOpenPanelResultListener()
{
}

void WebOpenPanelResultListener::didChooseFiles(const Vector<String>& files, const Vector<String>& replacementFiles)
{
    m_fileChooser->chooseFiles(files, replacementFiles);
}

void WebOpenPanelResultListener::didCancelFileChoosing()
{
    m_fileChooser->cancelFileChoosing();
}

#if PLATFORM(IOS_FAMILY)
void WebOpenPanelResultListener::didChooseFilesWithDisplayStringAndIcon(const Vector<String>& files, const String& displayString, WebCore::Icon* displayIcon)
{
    m_fileChooser->chooseMediaFiles(files, displayString, displayIcon);
}
#endif

} // namespace WebKit
