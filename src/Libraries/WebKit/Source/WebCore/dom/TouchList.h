/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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

#if ENABLE(IOS_TOUCH_EVENTS)
#include <WebKitAdditions/TouchListIOS.h>
#elif ENABLE(TOUCH_EVENTS)

#include "Node.h"
#include "Touch.h"
#include <wtf/FixedVector.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {

class TouchList : public RefCounted<TouchList> {
public:
    static Ref<TouchList> create()
    {
        return adoptRef(*new TouchList);
    }
    static Ref<TouchList> create(FixedVector<std::reference_wrapper<Touch>>&& touches)
    {
        return adoptRef(*new TouchList(WTFMove(touches)));
    }

    unsigned length() const { return m_values.size(); }

    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }
    Touch* item(unsigned);
    const Touch* item(unsigned) const;

    void append(Ref<Touch>&& touch) { m_values.append(WTFMove(touch)); }

private:
    TouchList() = default;

    explicit TouchList(FixedVector<std::reference_wrapper<Touch>>&& touches)
    {
        m_values = WTF::map(touches, [](auto& touch) -> Ref<Touch> {
            return touch.get();
        });
    }

    Vector<Ref<Touch>> m_values;
};

} // namespace WebCore

#endif // ENABLE(TOUCH_EVENTS)

