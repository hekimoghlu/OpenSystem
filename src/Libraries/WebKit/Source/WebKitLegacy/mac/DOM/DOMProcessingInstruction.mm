/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 1, 2023.
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
#import "DOMProcessingInstructionInternal.h"

#import "DOMNodeInternal.h"
#import "DOMStyleSheetInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/ProcessingInstruction.h>
#import <WebCore/StyleSheet.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::ProcessingInstruction*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMProcessingInstruction

- (NSString *)target
{
    WebCore::JSMainThreadNullState state;
    return IMPL->target();
}

- (DOMStyleSheet *)sheet
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->sheet()));
}

@end

WebCore::ProcessingInstruction* core(DOMProcessingInstruction *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::ProcessingInstruction*>(wrapper->_internal) : 0;
}

DOMProcessingInstruction *kit(WebCore::ProcessingInstruction* value)
{
    WebCoreThreadViolationCheckRoundOne();
    return static_cast<DOMProcessingInstruction*>(kit(static_cast<WebCore::Node*>(value)));
}

#undef IMPL
