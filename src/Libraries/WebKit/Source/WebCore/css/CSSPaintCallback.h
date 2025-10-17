/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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

#include "ActiveDOMCallback.h"
#include "CSSPaintSize.h"
#include "CallbackResult.h"
#include "StylePropertyMapReadOnly.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace JSC {
class JSValue;
} // namespace JSC

namespace WebCore {
class PaintRenderingContext2D;

class CSSPaintCallback : public RefCountedAndCanMakeWeakPtr<CSSPaintCallback>, public ActiveDOMCallback {
public:
    using ActiveDOMCallback::ActiveDOMCallback;

    virtual CallbackResult<void> handleEvent(JSC::JSValue, PaintRenderingContext2D&, CSSPaintSize&, StylePropertyMapReadOnly&, const Vector<String>&) = 0;
    virtual CallbackResult<void> handleEventRethrowingException(JSC::JSValue, PaintRenderingContext2D&, CSSPaintSize&, StylePropertyMapReadOnly&, const Vector<String>&) = 0;

    virtual ~CSSPaintCallback()
    {
    }

private:
    virtual bool hasCallback() const = 0;
};

} // namespace WebCore
