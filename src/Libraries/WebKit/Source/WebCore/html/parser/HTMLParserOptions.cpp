/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
#include "HTMLParserOptions.h"

#include "Document.h"
#include "FrameDestructionObserverInlines.h"
#include "FrameLoader.h"
#include "LocalFrame.h"
#include "ScriptController.h"
#include "Settings.h"
#include "SubframeLoader.h"

namespace WebCore {

HTMLParserOptions::HTMLParserOptions()
    : scriptingFlag(false)
    , usePreHTML5ParserQuirks(false)
    , maximumDOMTreeDepth(Settings::defaultMaximumHTMLParserDOMTreeDepth)
{
}

HTMLParserOptions::HTMLParserOptions(Document& document)
{
    RefPtr frame { document.frame() };
    if (document.settings().htmlParserScriptingFlagPolicy() == HTMLParserScriptingFlagPolicy::Enabled)
        scriptingFlag = true;
    else
        scriptingFlag = frame && frame->script().canExecuteScripts(ReasonForCallingCanExecuteScripts::NotAboutToExecuteScript) && document.allowsContentJavaScript();

    usePreHTML5ParserQuirks = document.settings().usePreHTML5ParserQuirks();
    maximumDOMTreeDepth = document.settings().maximumHTMLParserDOMTreeDepth();
}

}
