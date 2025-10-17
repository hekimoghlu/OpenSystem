/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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

#include <JavaScriptCore/JSCJSValueInlines.h>
#include <JavaScriptCore/JSObject.h>
#include <JavaScriptCore/Weak.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class GCObservation final : public RefCounted<GCObservation> {
public:
    template<typename... Args> static Ref<GCObservation> create(Args&&... args)
    {
        return adoptRef(*new GCObservation(std::forward<Args>(args)...));
    }

    bool wasCollected() const { return !m_observedValue; }

private:
    explicit GCObservation(JSC::JSObject*);

    mutable JSC::Weak<JSC::JSObject> m_observedValue;
};

} // namespace WebCore
